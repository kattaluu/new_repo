"""Joint Pauli readout and explicit forward/backward measurement policies."""

import pennylane as qml
import torch


MEASUREMENT_MODES = ("parameter_shift", "sampled_backprop", "analytic_train", "reinforce")


def effective_shots(config, training):
    """Evaluation includes validation/test; no_grad alone does not select evaluation mode."""
    if config.measurement_mode == "analytic_train" and training:
        return None
    return config.shots


def pauli_signs(device=None):
    indices = torch.arange(256, device=device)
    return torch.stack([1 - 2 * ((indices >> (7 - wire)) & 1) for wire in range(8)], dim=1)


def rotate_measurement_basis(measurement):
    """Ideal readout basis change, matching the existing X/Y observable convention."""
    for wire in range(8):
        if measurement == "Y":
            qml.adjoint(qml.S)(wires=wire)
        if measurement in ("X", "Y"):
            qml.Hadamard(wires=wire)


def sample_expectations(probabilities, shots, generator):
    """Sample joint bitstrings once per shot, preserving all eight-wire correlations (CPU)."""
    probabilities = probabilities.clamp_min(0)
    probabilities = probabilities / probabilities.sum(-1, keepdim=True)
    signs = pauli_signs(probabilities.device)
    total = torch.zeros((len(probabilities), 8), dtype=torch.long, device=probabilities.device)
    for start in range(0, shots, 4096):
        bitstrings = torch.multinomial(probabilities, min(shots - start, 4096), replacement=True, generator=generator)
        total += signs[bitstrings].sum(dim=1)
    return total.to(probabilities.dtype) / shots


def sample_reinforce_readout(probabilities, shots, generator):
    """Sample an N-shot readout and retain its differentiable joint log likelihood.

    The score is a SUM over shots, not a mean. Counts avoid retaining B*N indices.
    Probabilities are normalized consistently for sampling, scoring and the baseline.
    No gradient is propagated through the discrete readout itself.
    """
    probabilities = probabilities.clamp_min(0)
    probabilities = probabilities / probabilities.sum(-1, keepdim=True)
    with torch.no_grad():
        counts = torch.zeros_like(probabilities, dtype=torch.long)
        for start in range(0, shots, 4096):
            draws = torch.multinomial(probabilities.detach(), min(shots-start, 4096),
                                      replacement=True, generator=generator)
            counts.scatter_add_(1, draws, torch.ones_like(draws))
        signs = pauli_signs(probabilities.device).to(probabilities.dtype)
        sampled = counts.to(probabilities.dtype) @ signs / shots
        mean = probabilities.detach() @ signs
    # An unsampled zero-probability state contributes zero without log(0)/0*inf.
    selected = torch.where(counts > 0, probabilities, torch.ones_like(probabilities))
    score = (counts.to(probabilities.dtype) * selected.log()).sum(-1)
    return sampled, score, mean


def reinforce_readout(block, probabilities, shots, generator):
    sampled, score, mean = sample_reinforce_readout(probabilities, shots, generator)
    block._reinforce_score = score if torch.is_grad_enabled() and score.requires_grad else None
    block._reinforce_mean = mean
    return sampled


def reinforce_loss(losses, score, baseline):
    """Value-preserving likelihood-ratio objective for per-object finite-shot losses.

    Baseline must be independent of the current measurement draws. Both the
    advantage and baseline are detached: head derivatives come from losses only.
    """
    if losses.ndim != 1 or losses.shape != score.shape or losses.shape != baseline.shape:
        raise ValueError("REINFORCE: losses, score и baseline должны иметь размер [batch]")
    correction = (losses.detach() - baseline.detach()) * (score - score.detach())
    return losses + correction.to(losses.dtype)


def classification_loss(model, logits, targets, *, reduction="mean"):
    """Cross entropy, with the correct loss-aware REINFORCE term when requested.

    The analytic-mean head loss is a sample-independent control variate. It is
    detached and reuses the existing probability calculation, not another simulation.
    Returns the actual sampled CE value; only its backward estimate is modified.
    """
    if reduction not in ("mean", "sum", "none"):
        raise ValueError("reduction: mean, sum или none")
    quantum = model.quantum
    score = getattr(quantum, "_reinforce_score", None)
    active = (model.use_quantum and model.config.measurement_mode == "reinforce"
              and model.config.shots is not None and score is not None and torch.is_grad_enabled())
    if not active:
        return torch.nn.functional.cross_entropy(logits, targets, reduction=reduction)
    losses = torch.nn.functional.cross_entropy(logits, targets, reduction="none")
    batch_norms = [module for module in model.head.modules()
                   if isinstance(module, torch.nn.modules.batchnorm._BatchNorm) and module.training]
    saved_statistics = [(module, module.running_mean, module.running_var) for module in batch_norms]
    try:
        # A control variate must not update running statistics a second time.
        for module, running_mean, running_var in saved_statistics:
            module.training = False
            if module.track_running_stats:
                # The just-updated head statistics depend on the current sampled
                # readout. Use fixed neutral statistics for a draw-independent baseline.
                module.running_mean = torch.zeros_like(running_mean)
                module.running_var = torch.ones_like(running_var)
        with torch.no_grad():
            mean = quantum._reinforce_mean.to(dtype=next(model.head.parameters()).dtype)
            baseline = torch.nn.functional.cross_entropy(model.classify_readout(mean), targets, reduction="none")
    finally:
        for module, running_mean, running_var in saved_statistics:
            module.running_mean, module.running_var = running_mean, running_var
            module.training = True
    if batch_norms:
        # A training head with BN couples objects: use the whole-microbatch loss
        # as the score-function reward, rather than an individual object's loss.
        advantage = losses.detach().sum() - baseline.detach().sum()
        losses = losses + advantage * (score - score.detach()).to(losses.dtype)
    else:
        losses = reinforce_loss(losses, score, baseline)
    return losses if reduction == "none" else losses.mean() if reduction == "mean" else losses.sum()


class _AnalyticJacobian(torch.autograd.Function):
    @staticmethod
    def forward(ctx, mean, sampled):
        # Preserve the actual sample values, avoiding floating-point cancellation in mu+(sample-mu).detach().
        return sampled

    @staticmethod
    def backward(ctx, upstream):
        return upstream, None


def sampled_with_analytic_jacobian(mean, sampled):
    """Explicit surrogate: exact finite-shot forward, analytic d(mean)/d(parameters) backward.

    This is generally biased for the expected finite-shot nonlinear loss.
    It is NOT a parameter-shift or a score-function estimator.
    """
    return _AnalyticJacobian.apply(mean, sampled)


def readout_with_analytic_jacobian(probabilities, shots, generator):
    mean = probabilities @ pauli_signs(probabilities.device).to(probabilities.dtype)
    with torch.no_grad():
        sampled = sample_expectations(probabilities.detach(), shots, generator)
    return sampled_with_analytic_jacobian(mean, sampled)


def measurement_description(config):
    if config.shots is None:
        return "Аналитические средние, backprop (shots=None)"
    labels = {
        "parameter_shift": "Выборочные измерения и parameter-shift при обучении",
        "sampled_backprop": "Выборочный forward, аналитический Jacobian (суррогатный градиент)",
        "analytic_train": "Аналитическое обучение; конечные shots при валидации и тестировании",
        "reinforce": "REINFORCE: градиент ожидаемой кросс-энтропии конечных shots; аналитический baseline",
    }
    return f"{labels[config.measurement_mode]}; shots={config.shots}; seed={config.measurement_seed}"

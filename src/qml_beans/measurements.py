"""Joint Pauli readout and explicit forward/backward measurement policies."""

import pennylane as qml
import torch


MEASUREMENT_MODES = ("parameter_shift", "sampled_backprop", "analytic_train")


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
    }
    return f"{labels[config.measurement_mode]}; shots={config.shots}; seed={config.measurement_seed}"

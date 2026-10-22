import numpy as np
import pandas as pd
from scipy.stats import t

def sample_statistics(values, *, confidence=0.95, allow_ci=True):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not len(values) or (not np.isfinite(values).all()):
        raise ValueError('Statistics require a nonempty finite vector')
    if not 0 < confidence < 1:
        raise ValueError('confidence must lie between 0 and 1')
    n = len(values)
    result = dict(n=n, mean=float(values.mean()), sd=None, sem=None, ci_low=None, ci_high=None, ci_half_width=None)
    if n > 1:
        sd = float(values.std(ddof=1))
        result.update(sd=sd)
        if allow_ci:
            sem = sd / np.sqrt(n)
            half = float(t.ppf((1 + confidence) / 2, n - 1) * sem)
            result.update(sem=float(sem), ci_low=result['mean'] - half, ci_high=result['mean'] + half, ci_half_width=half)
    return result

def summarize_seed_runs(config, rows):
    summary = []
    for stage in config['stages']:
        selected = [row for row in rows if row['stage'] == stage['name']]
        for metric in ('test_accuracy', 'test_balanced_accuracy', 'test_macro_f1', 'test_loss', 'val_accuracy'):
            if not any((metric in row for row in selected)):
                continue
            stats = sample_statistics([row[metric] for row in selected], allow_ci=config['split_policy'] == 'fixed')
            summary.append(dict(stage=stage['name'], metric=metric, **stats))
    return summary

def _curves(rows, *, allow_ci=True):
    records = [dict(seed=row['seed'], **epoch) for row in rows for epoch in row['history']]
    if not records:
        return pd.DataFrame()
    frame = pd.DataFrame(records)
    records = []
    for epoch, selected in frame.groupby('epoch', sort=True):
        row = dict(epoch=int(epoch), n=int(len(selected)))
        for metric in ('train_loss', 'val_loss', 'test_loss', 'train_accuracy', 'val_accuracy', 'test_accuracy'):
            if metric not in selected:
                continue
            stats = sample_statistics(selected[metric], allow_ci=allow_ci)
            for name in ('mean', 'sd', 'ci_low', 'ci_high'):
                row[f'{metric}_{name}'] = stats[name]
        records.append(row)
    return pd.DataFrame(records)

def _plot_seed_report(config, rows, summary, output, band, show, *, include_validation=False, vision_style=False, legend_loc='lower right', legend_on='right', legend_ncol=1, panel_positions=((-0.12, 1.04), (-0.12, 1.04))):
    if legend_on not in ('left', 'right', 'both', 'none'):
        raise ValueError('legend_on: left / right / both / none')
    if type(legend_ncol) is not int or legend_ncol < 1:
        raise ValueError('legend_ncol: положительное целое')
    if vision_style:
        import matplotlib as mpl
        import matplotlib.style as mpl_style
        from .plot_style import vision_rc
        with mpl_style.context('seaborn-v0_8-whitegrid'), mpl.rc_context(vision_rc()):
            return _plot_seed_report_impl(config, rows, summary, output, band, show, include_validation=include_validation, vision_style=True, legend_loc=legend_loc, legend_on=legend_on, legend_ncol=legend_ncol, panel_positions=panel_positions)
    return _plot_seed_report_impl(config, rows, summary, output, band, show, include_validation=include_validation)

def _plot_seed_report_impl(config, rows, summary, output, band, show, *, include_validation=False, vision_style=False, legend_loc='lower right', legend_on='right', legend_ncol=1, panel_positions=((-0.12, 1.04), (-0.12, 1.04))):
    from .plot_style import PURPLE, validate_panel_positions, add_panel_label
    panel_positions = validate_panel_positions(panel_positions)
    from matplotlib.ticker import MaxNLocator
    if show:
        import matplotlib.pyplot as plt
        new_figure = plt.subplots
    else:
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure

        def new_figure(nrows, ncols, **kwargs):
            figure = Figure(**kwargs)
            FigureCanvasAgg(figure)
            return (figure, figure.subplots(nrows, ncols))
    error_name = 'sd' if band == 'sd' else 'ci_half_width'
    band_label = 'Среднее ± выборочное стандартное отклонение' if band == 'sd' else 'Среднее ± приближённый 95%-й ДИ (Стьюдент)'
    if len({row['seed'] for row in rows}) == 1:
        band_label = 'Среднее, один запуск (разброс не определён)'
    stage_names = {'hybrid': 'Гибридная модель', 'classical': 'Классическая модель', 'quantum_only': 'Обучение квантовой части', 'joint': 'Совместное обучение'}
    stage_labels = [stage_names.get(stage['name'], f'Этап {index + 1}') for index, stage in enumerate(config['stages'])]
    metric_labels = ('Вневыборочная точность (%)', 'Вневыборочная макро-F1 (%)') if config.get('cv_folds') else ('Точность на тестовой выборке (%)', 'Макро-F1 на тестовой выборке (%)')
    cv_note = f"; число блоков: {config['cv_folds']}" if config.get('cv_folds') else ''
    splits = [('test', 'Тестовая выборка', 'metrics_spread')]
    if include_validation:
        splits.append(('val', 'Валидационная выборка', 'metrics_spread_validation'))
    for prefix, split_label, filename in splits:
        fig, axes = new_figure(1, 2, figsize=(12, 5) if vision_style else (11, 4.5), constrained_layout=not vision_style)
        labels = ('Точность (%)', 'Макро-F1 (%)') if include_validation else metric_labels
        for axis, suffix, label in zip(axes, ('accuracy', 'macro_f1'), labels):
            metric = f'{prefix}_{suffix}'
            for index, stage in enumerate(config['stages']):
                selected = [row for row in rows if row['stage'] == stage['name']]
                stats = next((row for row in summary if row['stage'] == stage['name'] and row['metric'] == metric))
                offsets = np.linspace(-0.1, 0.1, len(selected)) if len(selected) > 1 else [0]
                axis.scatter(index + np.asarray(offsets), [100 * row[metric] for row in selected], color=PURPLE if vision_style else 'tab:blue', alpha=0.65, s=30, label='Отдельные запуски' if index == 0 else None)
                error = stats[error_name]
                axis.errorbar(index, 100 * stats['mean'], yerr=None if error is None else 100 * error, fmt='D', color=PURPLE if vision_style else 'tab:orange', capsize=6, markersize=6, zorder=5, label=band_label if index == 0 else None)
            axis.set_xticks(range(len(config['stages'])), stage_labels, rotation=15)
            axis.set(ylabel=label, title=label)
            if vision_style:
                add_panel_label(axis, 0 if axis is axes[0] else 1, panel_positions)
            axis.grid(axis='y', alpha=0.6 if vision_style else 0.25, linestyle=':' if vision_style else '-')
            if not vision_style:
                axis.legend(fontsize=8)
            elif legend_loc is not None and (legend_on == 'both' or legend_on == ('left' if axis is axes[0] else 'right')):
                axis.legend(loc=legend_loc, ncol=legend_ncol, fontsize=14)
        split_note = f'\n{split_label}' if include_validation else ''
        if not config.get('mixed_protocols'):
            fig.suptitle(f"Независимые запуски обучения: {len({row['seed'] for row in rows})}{cv_note}{split_note}")
        if vision_style:
            fig.tight_layout()
        fig.savefig(output / f'{filename}.png', dpi=600 if vision_style else 160)
        fig.savefig(output / f'{filename}.svg')
        if show:
            plt.show()
            plt.close(fig)
    for stage, stage_label in zip(config['stages'], stage_labels):
        selected = [row for row in rows if row['stage'] == stage['name']]
        curves = _curves(selected, allow_ci=config['split_policy'] == 'fixed')
        curves.to_csv(output / f"curves_{stage['name']}.csv", index=False)
        if curves.empty:
            continue
        fig, axes = new_figure(1, 2, figsize=(12, 5) if vision_style else (14, 3.8), constrained_layout=not vision_style)
        for axis, metrics, title in ((axes[0], ('train_loss', 'val_loss', 'test_loss'), 'Кросс-энтропия'), (axes[1], ('train_accuracy', 'val_accuracy', 'test_accuracy'), 'Точность')):
            for metric in metrics:
                if f'{metric}_mean' not in curves:
                    continue
                mean = curves[f'{metric}_mean'].to_numpy(dtype=float)
                metric_label = {'train': 'Обучающая выборка', 'val': 'Валидационная выборка', 'test': 'Тестовая выборка'}[metric.split('_')[0]]
                if vision_style:
                    prefix = metric.split('_')[0]
                    if prefix == 'train':
                        metric_label = 'Обучение'
                    if prefix == 'test':
                        metric_label = 'Тест'
                    line, = axis.plot(curves.epoch, mean, label=metric_label, color=PURPLE, linewidth=2, linestyle={'train': '-', 'test': '--', 'val': ':'}[prefix])
                else:
                    line, = axis.plot(curves.epoch, mean, label=metric_label, marker='o', markersize=3)
                if band == 'sd':
                    delta = curves[f'{metric}_sd'].to_numpy(dtype=float)
                    low, high = (mean - delta, mean + delta)
                else:
                    low = curves[f'{metric}_ci_low'].to_numpy(dtype=float)
                    high = curves[f'{metric}_ci_high'].to_numpy(dtype=float)
                axis.fill_between(curves.epoch, low, high, where=curves.n.to_numpy() >= 2, alpha=(0.15 if metric.startswith('train') else 0.1) if vision_style else 0.18, color=line.get_color())
            if vision_style:
                loss = metrics[0] == 'train_loss'
                axis.set(title='Функция потерь (Loss)' if loss else 'Точность (Accuracy)', xlabel='Эпоха обучения', ylabel='Значение кросс-энтропии' if loss else 'Значение точности')
                add_panel_label(axis, 0 if loss else 1, panel_positions)
                axis.set_xlim(left=0)
            else:
                axis.set(title=title, xlabel='Эпоха')
            axis.xaxis.set_major_locator(MaxNLocator(integer=True))
            axis.grid(alpha=0.6 if vision_style else 0.2, linestyle=':' if vision_style else '-')
            if vision_style:
                if legend_loc is not None and (legend_on == 'both' or legend_on == ('left' if axis is axes[0] else 'right')):
                    axis.legend(loc=legend_loc, ncol=legend_ncol, frameon=True)
            else:
                axis.legend()
        cv_note = '; в каждом запуске усреднены доступные блоки' if config.get('cv_folds') else ''
        if vision_style:
            fig.tight_layout()
        else:
            fig.suptitle(f'{stage_label}: {band_label}{cv_note}')
        fig.savefig(output / f"learning_curves_{stage['name']}.png", dpi=600 if vision_style else 160)
        fig.savefig(output / f"learning_curves_{stage['name']}.svg")
        if show:
            plt.show()
            plt.close(fig)

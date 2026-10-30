"""Passive local events and derived, identity-preserving result table exports.

One logger owns an event directory at a time. Events are immutable JSON files,
committed with the shared atomic writer; local records need no remote package.
"""
from datetime import datetime, timezone
from pathlib import Path

from .storage import atomic_write_json


class LocalLogger:
    """Single-writer event journal, reopened without replacing earlier records.

    An event is diagnostic only: it never commits an evaluation or a policy.
    Each record is flushed/fsynced before returning. Partial temporary files
    from interrupted writes are ignored on reopen.
    """

    # Params: self, directory; returns: None.
    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        indices = [int(p.stem) for p in self.directory.glob('*.json') if p.stem.isdigit()]
        self._next_index = max(indices, default=-1) + 1
        self._closed = False

    # Params: self, event, identity, payload; returns: None.
    def record(self, event, identity, payload):
        if self._closed:
            raise RuntimeError('LocalLogger is closed')
        if not isinstance(event, str) or not event.strip():
            raise ValueError('event must be a nonempty string')
        path = self.directory / f'{self._next_index:012d}.json'
        if path.exists():
            raise FileExistsError(path)
        atomic_write_json(path, {
            'schema_version': 1, 'event_index': self._next_index,
            'event': event, 'identity': identity, 'payload': payload,
            'created_utc': datetime.now(timezone.utc).isoformat(),
        })
        self._next_index += 1

    def flush(self):
        # Every successful record is already durable at the file level.
        pass

    # Params: self; returns: None.
    def close(self):
        self.flush()
        self._closed = True

    # Params: self; returns: self.
    def __enter__(self):
        return self

    # Params: self, *_; returns: None.
    def __exit__(self, *_):
        self.close()


from contextlib import contextmanager


@contextmanager
def _external_state_guard():
    """Preserve existing learner RNG without importing Torch just for logging."""
    import sys
    if 'torch' in sys.modules:
        from .evaluation import evaluation_state_guard
        with evaluation_state_guard():
            yield
        return
    import random
    python_state = random.getstate()
    numpy = sys.modules.get('numpy')
    numpy_state = numpy.random.get_state() if numpy is not None else None
    try:
        yield
    finally:
        random.setstate(python_state)
        if numpy_state is not None:
            numpy.random.set_state(numpy_state)


class WandbLogger:
    """Optional run-scoped event mirror; never uploads models or marks completion.

    SDK logging is queued; flush does not claim a remote durability barrier.
    finish at close drains the SDK queue. LocalLogger remains authoritative.
    """

    # Params: self, run; returns: None.
    def __init__(self, run):
        if not callable(getattr(run, 'log', None)) or not callable(getattr(run, 'finish', None)):
            raise TypeError('External client init must return a run with log and finish methods')
        self.run = run
        self._failed = False
        self._closed = False

    # Params: self, event, identity, payload; returns: None.
    def record(self, event, identity, payload):
        if self._closed:
            raise RuntimeError('WandbLogger is closed')
        if event in ('policy_failure', 'experiment_failure'):
            self._failed = True
        # Do not use actual policy transitions as SDK step: they restart for each
        # policy/replicate. Preserve those coordinates inside the identity.
        self.run.log({'event': event, 'identity': identity, 'payload': payload})

    # Params: self; returns: None.
    def flush(self):
        pass  # Run.log queues events; only finish provides the SDK's drain.

    # Params: self, failed; returns: None.
    def close(self, *, failed=False):
        if self._closed:
            return
        self._closed = True
        self.run.finish(exit_code=1 if failed or self._failed else 0)


class CompositeLogger:
    """Commit locally first; external failures disable only the optional mirror."""

    # Params: self, local, external; returns: None.
    def __init__(self, local, external=None):
        self.local = local
        self.external = external
        self.directory = local.directory
        self._closed = False
        self._local_failed = False

    # Params: self, operation, error; returns: None.
    def _warning(self, operation, error):
        import warnings
        # Do not copy SDK exception text, which may contain credentials/URLs.
        self.local.record('external_logging_warning', None, {
            'operation': operation, 'error_type': type(error).__name__,
            'external_disabled': True, 'local_records_authoritative': True,
        })
        try:
            warnings.warn(f'External logging disabled after {operation} ({type(error).__name__}); '
                          'local records remain authoritative.', RuntimeWarning, stacklevel=3)
        except Warning:
            # A warning-as-error filter must not turn optional telemetry into a
            # training failure after the local warning has been committed.
            pass

    # Params: self, operation, error; returns: None.
    def _disable(self, operation, error):
        external, self.external = self.external, None
        try:
            self._warning(operation, error)
        finally:
            if external is not None and operation != 'close':
                try:
                    with _external_state_guard():
                        external.close(failed=True)
                except Exception as cleanup_error:
                    self._warning('close', cleanup_error)

    # Params: self, operation, *args, **kwargs; returns: None.
    def _remote(self, operation, *args, **kwargs):
        if self.external is None:
            return
        try:
            with _external_state_guard():
                getattr(self.external, operation)(*args, **kwargs)
        except Exception as error:
            self._disable(operation, error)

    # Params: self, event, identity, payload; returns: None.
    def record(self, event, identity, payload):
        if self._closed:
            raise RuntimeError('CompositeLogger is closed')
        try:
            self.local.record(event, identity, payload)
        except BaseException:
            self._local_failed = True
            raise
        if self.external is not None:
            try:
                from .storage import _json_value
                # Give optional code its own primitive copies, never borrowed
                # learner arrays or identity records.
                values = (_json_value(identity), _json_value(payload))
            except Exception as error:
                self._disable('serialization', error)
            else:
                self._remote('record', event, *values)

    # Params: self; returns: None.
    def flush(self):
        if self._closed:
            return
        try:
            self.local.flush()
        except BaseException:
            self._local_failed = True
            raise
        self._remote('flush')

    # Params: self; returns: None.
    def close(self):
        if self._closed:
            return
        try:
            self.flush()
        finally:
            try:
                self._remote('close', failed=self._local_failed)
                self.external = None
            finally:
                try:
                    self.local.close()
                finally:
                    self._closed = True

    # Params: self; returns: self.
    def __enter__(self):
        if self._closed:
            raise RuntimeError('CompositeLogger is closed')
        return self

    # Params: self, error_type, *_; returns: None.
    def __exit__(self, error_type, *_):
        if error_type is not None:
            self._local_failed = True
        self.close()


def make_logger(config, experiment_dir, *, external_client=None):
    """Select off/offline/online explicitly; import W&B only on enabled requests.

    Enabled modes require owner-supplied project and entity, never SDK/environment
    defaults. external_client is an injected SDK-compatible object for tests.
    Dependency/SDK failures degrade to local events with a persisted warning;
    invalid settings fail before directory creation, and local I/O errors propagate.
    """
    from .types import ExperimentConfig
    if not isinstance(config, ExperimentConfig):
        raise TypeError('config must be ExperimentConfig')
    settings = config.logging
    if settings.mode not in ('off', 'offline', 'online'):
        raise ValueError('Unsupported external logging mode')
    if settings.mode != 'off' and (not settings.project or not settings.project.strip()
                                   or not settings.entity or not settings.entity.strip()):
        raise ValueError('External logging requires owner-supplied project and entity')
    local = LocalLogger(Path(experiment_dir) / 'logs' / 'events')
    if settings.mode == 'off':
        return local
    composite = CompositeLogger(local)
    run = None
    try:
        with _external_state_guard():
            from .storage import _json_value
            if external_client is None:
                from importlib import import_module
                external_client = import_module('wandb')
            external_directory = Path(experiment_dir) / 'logs' / 'external'
            external_directory.mkdir(parents=True, exist_ok=True)
            run = external_client.init(project=settings.project, entity=settings.entity,
                mode=settings.mode, dir=str(external_directory.absolute()), name=config.experiment_id,
                config=_json_value(config), reinit='create_new', settings={'init_timeout': 20, 'quiet': True})
            composite.external = WandbLogger(run)
    except Exception as error:
        try:
            try:
                composite._warning('initialization', error)
            finally:
                if run is not None and callable(getattr(run, 'finish', None)):
                    try:
                        with _external_state_guard():
                            run.finish(exit_code=1)
                    except Exception as cleanup_error:
                        composite._warning('close', cleanup_error)
        except BaseException:
            local.close()
            raise
    except BaseException:
        try:
            if run is not None and callable(getattr(run, 'finish', None)):
                try:
                    with _external_state_guard():
                        run.finish(exit_code=1)
                except Exception:
                    pass  # Preserve cancellation; never convert it to telemetry success.
        finally:
            local.close()
        raise
    return composite



def _write_csv(path, columns, rows):
    """Stream UTF-8 rows to an atomic sibling file; blanks represent nulls."""
    import csv
    import io
    from .storage import _atomic_write, _json_value
    count = 0
    # Params: stream; returns: None.
    def write(stream):
        nonlocal count
        text = io.TextIOWrapper(stream, encoding='utf-8', newline='')
        try:
            writer = csv.DictWriter(text, fieldnames=columns, lineterminator='\n')
            writer.writeheader()
            for row in rows:
                values = _json_value(row)
                writer.writerow({key: ('true' if value else 'false') if type(value) is bool else value
                                 for key, value in values.items()})
                count += 1
            text.flush()
        finally:
            text.detach()  # _atomic_write retains ownership of the binary stream.
    _atomic_write(path, write)
    return count


def _plot_path(save_path):
    """Validate the explicit format before creating a figure or directory."""
    path = Path(save_path)
    if path.suffix.lower() not in ('.png', '.pdf', '.svg'):
        raise ValueError('Plot save_path must end in .png, .pdf or .svg')
    return path


def _save_figure(figure, save_path):
    """Atomically export a standalone figure without pyplot/global backends."""
    from .storage import _atomic_write
    path = _plot_path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write(path, lambda stream: figure.savefig(stream, format=path.suffix[1:].lower(),
                                                     dpi=160, bbox_inches='tight'))


def plot_hypervolume_curve(hv_summary, actual_steps, save_path, *, reference=None):
    """Plot complete-seed validation HV with a sample-SD band and count panel.

    Preserve NaN gaps: no available-seed substitute, interpolation or line across
    an incomplete budget. The optional reference is explicitly in scaled reward
    space; the three-argument interface still labels its fixed-reference space.
    """
    import numpy as np
    from .types import HVSummary
    from .pareto import hypervolume_2d_max
    if not isinstance(hv_summary, HVSummary):
        raise TypeError('hv_summary must be HVSummary')
    steps = np.asarray(actual_steps)
    if (steps.dtype.kind not in 'iu' or steps.shape != hv_summary.actual_steps.shape
            or not np.array_equal(steps, hv_summary.actual_steps)):
        raise ValueError('actual_steps must exactly match the saved integer budget axis')
    if reference is not None:
        hypervolume_2d_max(np.empty((0, 2)), reference)  # Shared finite-reference validation.
    path = _plot_path(save_path)
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    figure = Figure(figsize=(9, 6.5))
    FigureCanvasAgg(figure)
    try:
        ax, counts = figure.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios': [3.3, 1]})
        mean, sd = hv_summary.mean, hv_summary.sd
        ax.plot(steps, mean, 'o-', color='#2166ac', label='Mean across all declared training seeds')
        ax.fill_between(steps, mean - sd, mean + sd, where=hv_summary.completeness,
                        color='#2166ac', alpha=.2, label='±1 sample SD across training seeds')
        ax.set_ylabel('Hypervolume (scaled reward space)')
        ax.set_title('Validation hypervolume at completed PPO updates')
        ax.grid(alpha=.25)
        ax.legend(loc='best', fontsize=9)
        label = ('Fixed scaled reference: ' + str(tuple(float(x) for x in reference))
                 if reference is not None else 'Fixed reference in scaled reward space')
        ax.text(.01, .02, label, transform=ax.transAxes, fontsize=9, va='bottom')
        if not hv_summary.completeness.any():
            ax.text(.5, .5, 'No budget has all declared training seeds complete',
                    transform=ax.transAxes, ha='center', va='center', fontsize=10)
        counts.step(steps, hv_summary.complete_seed_counts, where='mid', color='#555555')
        counts.plot(steps, hv_summary.complete_seed_counts, 'o', color='#555555', markersize=4)
        counts.set_ylim(-.1, hv_summary.declared_seed_count + .3)
        counts.set_yticks(sorted({0, hv_summary.declared_seed_count}))
        counts.set_ylabel(f'Complete seeds\n(of {hv_summary.declared_seed_count})')
        counts.set_xlabel('Actual training transitions per policy')
        counts.ticklabel_format(axis='x', style='plain', useOffset=False)
        counts.grid(alpha=.25)
        figure.tight_layout()
        _save_figure(figure, path)
    finally:
        figure.clear()


def _label_front_points(ax, groups):
    """Place exact-duplicate groups deterministically using rendered text bounds."""
    figure = ax.figure
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    from matplotlib.transforms import Bbox
    legend = ax.get_legend()
    occupied = [legend.get_window_extent(renderer).expanded(1.03, 1.08)] if legend else []
    for point in groups:
        x, y = ax.transData.transform(point)
        occupied.append(Bbox.from_bounds(x - 7, y - 7, 14, 14))
    bounds = ax.get_window_extent(renderer)
    offsets = [(x, y) for y in (10, -24, 30, -44, 50, -64, 75, -90, 105, -120)
               for x in (10, -10, 35, -35, 70, -70, 110, -110, 155, -155)]
    for point, labels in groups.items():
        annotation = ax.annotate('\n'.join(labels), xy=point, xytext=(10, 10), textcoords='offset points',
            fontsize=8, ha='left', va='bottom', annotation_clip=False,
            bbox={'boxstyle': 'round,pad=.2', 'facecolor': 'white', 'edgecolor': 'none', 'alpha': .85},
            arrowprops={'arrowstyle': '-', 'color': '.5', 'lw': .6})
        best, best_score = None, float('inf')
        for x, y in offsets:
            annotation.set_position((x, y))
            annotation.set_ha('left' if x >= 0 else 'right')
            annotation.set_va('bottom' if y >= 0 else 'top')
            # Text bbox excludes leader lines, which necessarily touch points.
            annotation.update_positions(renderer)
            box = annotation.get_bbox_patch()
            annotation.update_bbox_position_size(renderer)
            extent = box.get_window_extent(renderer).expanded(1.04, 1.12)
            outside = (max(0., bounds.x0 - extent.x0) + max(0., extent.x1 - bounds.x1)
                       + max(0., bounds.y0 - extent.y0) + max(0., extent.y1 - bounds.y1))
            score = 1000 * sum(extent.overlaps(other) for other in occupied) + outside
            if score < best_score:
                best, best_score = (x, y, extent), score
            if score == 0:
                break
        x, y, extent = best
        annotation.set_position((x, y))
        annotation.set_ha('left' if x >= 0 else 'right')
        annotation.set_va('bottom' if y >= 0 else 'top')
        occupied.append(extent)


def plot_final_front(final_results, seed_index, weights, reference, save_path, *, space='scaled'):
    """Plot one final test set, labelled by immutable preference index/value.

    Default to scaled reward space. For a separate raw-space diagnostic, pass
    space='raw' and the corresponding raw-space reference explicitly. Incomplete
    sets show available points with a warning, and no main front or HV.
    """
    import numpy as np
    from .types import FinalResults
    from .config import validate_weights
    from .pareto import unique_front_indices, hypervolume_2d_max
    if not isinstance(final_results, FinalResults):
        raise TypeError('final_results must be FinalResults')
    if type(seed_index) is not int or not 0 <= seed_index < len(final_results.validity):
        raise ValueError('seed_index must identify one declared training replicate')
    if space not in ('raw', 'scaled'):
        raise ValueError("space must be 'raw' or 'scaled'")
    weights = validate_weights(weights)
    if len(weights) != final_results.validity.shape[1]:
        raise ValueError('weights must retain every declared preference index')
    hypervolume_2d_max(np.empty((0, 2)), reference)
    reference = np.asarray(reference, dtype=np.float64)
    path = _plot_path(save_path)
    valid = final_results.validity[seed_index]
    points = getattr(final_results, 'means_' + space)[seed_index]
    complete = bool(valid.all())
    indices = unique_front_indices(points, range(len(weights))) if complete else ()
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    figure = Figure(figsize=(9.5, 7))
    FigureCanvasAgg(figure)
    try:
        ax = figure.subplots()
        scatter = ax.scatter(points[valid, 0], points[valid, 1], c=weights[valid, 0],
            cmap='viridis', vmin=0., vmax=1., s=65, edgecolors='#333333', linewidths=.6,
            label='Final policies (mean across test episodes)', zorder=3)
        if complete:
            ax.plot(points[indices, 0], points[indices, 1], '-', color='#222222', linewidth=1.5,
                    label='Nondominated final set', zorder=2)
        ax.scatter(*reference, marker='x', color='#a63603', s=65,
                   label=f'Fixed {space} reference ({reference[0]:g}, {reference[1]:g})', zorder=4)
        colorbar = figure.colorbar(scatter, ax=ax, pad=.025)
        colorbar.set_label('Tracking preference weight')
        ax.set_xlabel(f'Mean discounted tracking return ({space} reward space)')
        ax.set_ylabel(f'Mean discounted search return ({space} reward space)')
        ax.set_title(f'Final test reward front — training seed index {seed_index}')
        ax.grid(alpha=.25)
        ax.margins(x=.15, y=.2)
        ax.legend(loc='upper left', fontsize=8)
        status = (f'{len(weights)}/{len(weights)} policies complete; HV ({space}) = '
                  f'{hypervolume_2d_max(points, reference):.6g}' if complete else
                  f'Incomplete set: {int(valid.sum())}/{len(weights)} policies; main front and HV unavailable')
        figure.text(.5, .015, status, ha='center', fontsize=9)
        figure.tight_layout(rect=(0, .04, 1, 1))
        groups = {}
        for j in np.flatnonzero(valid):
            point = tuple(points[j].tolist())
            groups.setdefault(point, []).append(f'w{j} = ({weights[j, 0]:.3g}, {weights[j, 1]:.3g})')
        _label_front_points(ax, groups)
        _save_figure(figure, path)
    finally:
        figure.clear()


def plot_operational_metrics(results, weights, save_dir, *, actual_budget=None):
    """Plot separate physical diagnostics per seed at one declared budget.

    Intermediate results default to the final scheduled budget, never a selected
    best/last-available checkpoint. FinalResults use saved final test metrics.
    Equal-episode means/sample SD use available values; missing weights remain
    gaps and partial episode availability is explicitly marked. No search-time
    axis is inferred for PPO. Shared EpisodeMetrics values are not recomputed.
    """
    import numpy as np
    from .config import validate_weights
    from .types import ExperimentResults, FinalResults
    final = isinstance(results, FinalResults)
    if not final and not isinstance(results, ExperimentResults):
        raise TypeError('results must be ExperimentResults or FinalResults')
    if not final:
        results.validate()
        actual_budget = int(results.actual_budgets[-1]) if actual_budget is None else actual_budget
        if type(actual_budget) is not int or actual_budget not in results.actual_budgets:
            raise ValueError('actual_budget must identify a saved evaluation slot')
        k = list(results.actual_budgets).index(actual_budget)
    elif actual_budget is not None and (type(actual_budget) is not int or actual_budget < 1):
        raise ValueError('actual_budget must be a positive integer')
    weights = validate_weights(weights)
    if len(weights) != results.validity.shape[-1]:
        raise ValueError('weights must match every declared preference slot')
    order = np.argsort(weights[:, 0], kind='stable')
    panels = (
        ('mean_positional_error', 'Mean positional error', 'World units'),
        ('mean_final_positional_error', 'Final positional error', 'World units'),
        ('mean_covariance_trace', 'Mean position covariance trace', 'Squared world units'),
        ('unique_new_detections', 'Unique new detections', 'Targets per episode'),
        ('final_coverage', 'Final coverage', 'Raw coverage'),
        ('mean_coverage', 'Mean coverage', 'Raw coverage'),
        ('mean_known_observation_frequency', 'Initially known targets\nObservation frequency', 'Observations / eligible timestep'),
        ('mean_observation_frequency', 'Known + detected targets\nObservation frequency', 'Observations / eligible timestep'),
    )
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    folder = Path(save_dir)
    for i in range(results.validity.shape[0]):
        figure = Figure(figsize=(14, 10.5))
        FigureCanvasAgg(figure)
        try:
            axes = figure.subplots(3, 3).ravel()
            for ax, (name, title, unit) in zip(axes, panels):
                means = np.full(len(weights), np.nan); sds = means.copy()
                counts = np.zeros(len(weights), dtype=np.int64)
                expected = np.zeros(len(weights), dtype=np.int64)
                for j in range(len(weights)):
                    metrics = (results.operational_metrics[i][j] if results.operational_metrics else ()) if final else results.episode_metric_tables.get((i, k, j), ())
                    if not (results.validity[i, j] if final else results.validity[i, k, j]):
                        continue
                    expected[j] = len(metrics)
                    values = [getattr(m, name) for m in metrics if getattr(m, name) is not None]
                    counts[j] = len(values)
                    if values:
                        means[j] = np.mean(values)
                        sds[j] = np.std(values, ddof=1) if len(values) > 1 else 0.
                x, mean, sd = weights[order, 0], means[order], sds[order]
                ax.plot(x, mean, 'o-', color='#2166ac', markersize=4)
                ax.fill_between(x, mean - sd, mean + sd, where=np.isfinite(mean), color='#2166ac', alpha=.18)
                partial = (counts > 0) & (counts < expected)
                ax.scatter(weights[partial, 0], means[partial], s=65, facecolors='none',
                           edgecolors='#d95f02', linewidths=1.2, zorder=4)
                ax.set_title(title, fontsize=10)
                ax.set_ylabel(unit, fontsize=9)
                ax.set_xlabel('Tracking preference weight', fontsize=9)
                # Index labels retain saved identity even with arbitrary weight order.
                if len(weights) <= 12:
                    ax.set_xticks(x, [f'w{j}\n{weights[j, 0]:.2g}' for j in order], fontsize=7)
                ax.grid(alpha=.2)
                if not counts.any():
                    ax.text(.5, .5, 'No available episode metrics', ha='center', va='center',
                            transform=ax.transAxes, fontsize=9)
            axes[8].axis('off')
            axes[8].text(.02, .95,
                'Mean ±1 sample SD across available episodes.\n\n'
                'Orange rings: some episode values unavailable.\n'
                'Missing policies/metrics remain gaps.\n\n'
                'Initially known frequencies count from t0.\n'
                'Detected unknowns count from first detection;\n'
                'never-detected unknowns stay undefined.\n\n'
                'PPO search time is not plotted: no explicit\n'
                'search/track controller modes.\n\n'
                'Sample counts and exposure denominators:\n'
                'see the accompanying report tables.',
                va='top', fontsize=9, linespacing=1.4)
            phase = 'Final test' if final else 'Intermediate validation'
            budget_label = '' if actual_budget is None else f'; actual budget {actual_budget}'
            figure.suptitle(f'{phase} physical metrics — training seed index {i}{budget_label}', fontsize=13)
            figure.tight_layout(rect=(0, 0, 1, .96), h_pad=2., w_pad=2.)
            _save_figure(figure, folder / f'operational_seed_{i}.png')
        finally:
            figure.clear()


def _load_report_results(root, config):
    """Read canonical records only; do not read caches, load models or evaluate."""
    import numpy as np
    from . import storage
    from .types import ExperimentResults, FinalResults
    from .pareto import compute_hv_curve, unique_front_indices, hypervolume_2d_max
    root, payload, fingerprint = storage._evaluation_context(root, config)
    s, w, f = len(config.run_seeds), config.n_weights, config.final_episodes
    results = ExperimentResults.empty(s, payload['budget_plan']['eval_steps'], w, config.eval_episodes)
    raw = np.full((s, w, f, 2), np.nan, np.float64); scaled = np.full_like(raw, np.nan)
    means_raw = np.full((s, w, 2), np.nan, np.float64); means_scaled = np.full_like(means_raw, np.nan)
    valid = np.zeros((s, w), bool); metrics = [[() for _ in range(w)] for _ in range(s)]
    for namespace in ('evaluations', 'final'):
        for path in sorted((root / 'results' / namespace).glob('seed_*/w_*/step_*.json')):
            record, meta = storage._load_evaluation_record(path)
            expected, kind = storage._evaluation_location(root, payload, fingerprint, record.identity, record)
            if path != expected or kind != namespace or meta['kind'] != namespace:
                raise storage.StorageError('Misplaced report source record: ' + str(path))
            i, j = record.identity.run_index, record.identity.weight_index
            if namespace == 'evaluations':
                k = payload['budget_plan']['eval_steps'].index(record.actual_step)
                storage.store_eval_slot(results, i, k, j, record)
            else:
                raw[i, j] = record.returns_raw; scaled[i, j] = record.returns_scaled
                means_raw[i, j] = record.mean_raw; means_scaled[i, j] = record.mean_scaled
                valid[i, j] = True; metrics[i][j] = record.episode_metrics
    results.hypervolume[:] = compute_hv_curve(results, config.reference_scaled)
    fronts = tuple(tuple(int(j) for j in unique_front_indices(means_scaled[i], range(w))) if valid[i].all() else () for i in range(s))
    hv = np.asarray([hypervolume_2d_max(means_scaled[i], config.reference_scaled) if valid[i].all() else np.nan for i in range(s)])
    final = FinalResults(means_raw, means_scaled, raw, scaled, valid, tuple(payload['seed_plans'][0]['final_seeds']),
                         fronts, hv, tuple(tuple(row) for row in metrics))
    _table_slots(results, final, root, payload, fingerprint)  # Includes resume/journal validation.
    return results, final, payload, fingerprint


def _report_input_hashes(root):
    """Snapshot initialization, committed evaluations and current resume journals."""
    import hashlib
    paths = [root / name for name in ('config.json', 'manifest.json', 'weights.npy', 'seed_plan.json', 'budget_plan.json')]
    for namespace in ('evaluations', 'final'):
        for path in (root / 'results' / namespace).glob('seed_*/w_*/step_*.json'):
            paths.extend((path, path.with_suffix('.npz')))
    paths.extend((root / 'results/evaluations').glob('seed_*/w_*/RESUME.json'))
    return {path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(paths)}


def generate_reports(experiment_dir, config):
    """Regenerate tables, reward/physical figures and a hashed report manifest.

    Read committed local evaluations only, preserving absent slots. A report is
    descriptive and never marks training/test completion. Publish its manifest
    last and reject inputs changed during generation; consumers verify hashes.
    No worker, policy loading, new episode or external logger is involved.
    """
    import hashlib
    import numpy as np
    from . import storage
    from .types import ReportManifest
    from .pareto import summarize_hv_across_seeds
    root, _, _ = storage._evaluation_context(experiment_dir, config)
    before = _report_input_hashes(root)
    results, final, payload, fingerprint = _load_report_results(root, config)
    weights = np.load(root / 'weights.npy', allow_pickle=False)
    target = payload['budget_plan']['effective_total']
    summary = summarize_hv_across_seeds(results.hypervolume, results.actual_budgets)
    relative = ['reports/tables/' + name for name in (
        'policy_results.csv', 'episode_returns.csv', 'episode_metrics.csv', 'target_observation_metrics.csv',
        'operational_summary.csv', 'hypervolume.csv', 'table_metadata.json')]
    relative += ['reports/plots/hypervolume.png', 'reports/report_inputs.json', 'reports/report.md']
    for i in range(len(config.run_seeds)):
        relative.extend((f'reports/plots/final_seed_{i}_scaled.png', f'reports/plots/final_seed_{i}_raw.png',
            f'reports/plots/validation_operational/operational_seed_{i}.png',
            f'reports/plots/final_operational/operational_seed_{i}.png'))
    for name in relative + ['reports/report_manifest.json']:
        (root / name).resolve().relative_to(root)
    with np.errstate(over='ignore', invalid='ignore'):
        raw_reference = np.asarray(config.reference_scaled, np.float64) * np.asarray(config.objective_scales, np.float64)
    if not np.isfinite(raw_reference).all():
        raise storage.StorageError('Scaled-to-raw reference conversion is nonfinite')
    write_summary_tables(results, final, root)
    plot_hypervolume_curve(summary, results.actual_budgets, root / 'reports/plots/hypervolume.png', reference=config.reference_scaled)
    for i in range(len(config.run_seeds)):
        plot_final_front(final, i, weights, config.reference_scaled, root / f'reports/plots/final_seed_{i}_scaled.png')
        plot_final_front(final, i, weights, raw_reference, root / f'reports/plots/final_seed_{i}_raw.png', space='raw')
    plot_operational_metrics(results, weights, root / 'reports/plots/validation_operational', actual_budget=target)
    plot_operational_metrics(final, weights, root / 'reports/plots/final_operational', actual_budget=target)
    report = ['# Multi-objective PPO experiment report', '', f'Experiment fingerprint: `{fingerprint}`.', '',
        f'Declared training seeds: {list(config.run_seeds)}. Preferences per seed: {config.n_weights}.',
        f'Requested transitions per policy: {config.requested_timesteps_per_policy}; effective budget: {target}.',
        f'Objective order: tracking, search. Scales: {config.objective_scales}; scaled reference: {config.reference_scaled}.', '',
        'This report describes committed evaluations; it does not establish policy training completion.',
        'Validation and final-test episodes remain separate. Fronts are per training seed.', '',
        '## Validation hypervolume completeness', '', '| Actual budget | Complete training seeds |', '| --- | --- |']
    report.extend(f'| {int(step)} | {int(count)}/{len(config.run_seeds)} |' for step, count in zip(results.actual_budgets, summary.complete_seed_counts))
    report.extend(('', '[Hypervolume mean and sample SD](plots/hypervolume.png)', '', '## Per-seed final and physical metrics', ''))
    for i, seed in enumerate(config.run_seeds):
        report.extend((f'- Seed index {i}, training seed {seed}: {int(final.validity[i].sum())}/{config.n_weights} final evaluations.',
            f'  [Scaled reward front](plots/final_seed_{i}_scaled.png) · [Raw reward front](plots/final_seed_{i}_raw.png) · '
            f'[Validation physical metrics](plots/validation_operational/operational_seed_{i}.png) · '
            f'[Final-test physical metrics](plots/final_operational/operational_seed_{i}.png)'))
    report.extend(('', '## Tables and interpretation', '',
        '[Reward summaries](tables/policy_results.csv) · [Episode metrics](tables/episode_metrics.csv) · '
        '[Target exposure](tables/target_observation_metrics.csv) · [Operational sample counts](tables/operational_summary.csv)', '',
        'Operational charts show equal-episode means and sample SD over available values. Missing values remain gaps; '
        'orange rings mark incomplete episode availability. Known-target frequencies count from t0; '
        'the known-plus-detected average excludes never-detected unknowns. PPO search time is undefined without controller modes and is not plotted.', '',
        'Within-episode positional/covariance pair counts were not retained; table cells contain explicit missing reasons. '
        'Reward objectives remain the recorded vector returns, separate from physical metrics.', '',
        'Verify report_manifest.json hashes and report_inputs.json before using regenerated outputs. '
        'Aggregate NPY/NPZ caches and uncommitted numeric or temporary files were not used.'))
    storage._atomic_write(root / 'reports/report.md', lambda stream: stream.write(('\n'.join(report) + '\n').encode('utf-8')))
    atomic_write_json(root / 'reports/report_inputs.json', {
        'schema_version': 1, 'experiment_fingerprint': fingerprint, 'input_hashes': before,
        'report_generator_source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'operational_actual_budget': target, 'final_complete_seed_count': int(final.validity.all(axis=1).sum()),
        'declared_seed_count': len(config.run_seeds), 'normalization': payload['config']['normalization'],
        'objective_scales': config.objective_scales, 'reference_scaled': config.reference_scaled,
        'source': 'Canonical committed JSON/NPZ evaluations; no model loading or new evaluation',
    })
    if _report_input_hashes(root) != before:
        raise storage.StorageError('Report inputs changed during generation; retry after writers finish')
    manifest = ReportManifest(tuple(relative),
        {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in relative}, fingerprint,
        tuple(int(step) for step in results.actual_budgets), tuple(int(n) for n in summary.complete_seed_counts),
        datetime.now(timezone.utc).isoformat())
    atomic_write_json(root / 'reports/report_manifest.json', manifest)
    return manifest


def _table_slots(results, final_results, root, payload, fingerprint):
    """Validate supplied tensors against canonical committed evaluation records."""
    import numpy as np
    from . import storage
    from .types import ExperimentResults, FinalResults
    if not isinstance(results, ExperimentResults):
        raise TypeError('results must be ExperimentResults')
    results.validate()
    science, budget = payload['config'], payload['budget_plan']
    shape = (len(science['run_seeds']), len(budget['eval_steps']), len(payload['weights']['values']))
    if results.validity.shape != shape or not np.array_equal(results.actual_budgets, budget['eval_steps']):
        raise storage.StorageError('Intermediate table axes must match the saved experiment')
    if final_results is not None and (not isinstance(final_results, FinalResults)
            or final_results.validity.shape != (shape[0], shape[2])
            or final_results.test_seeds != tuple(payload['seed_plans'][0]['final_seeds'])):
        raise storage.StorageError('Final table axes/test seeds must match the saved experiment')
    attempts = []
    for i in range(shape[0]):
        for j in range(shape[2]):
            journal = root / 'results/evaluations' / f'seed_{i}' / f'w_{j}' / 'RESUME.json'
            state = None
            if journal.exists():
                plan = storage._read_json(journal)
                state = storage._validate_resume_plan(root, plan, payload, fingerprint, i, j)
                if plan['status'] != 'ready':
                    raise storage.StorageError('Cannot export while resume reconciliation is pending')
            attempts.append({'run_index': i, 'weight_index': j,
                'state': None if state is None else storage._json_value(state)})
    slots = []
    phases = [('intermediate', results, budget['eval_steps'])]
    if final_results is not None:
        phases.append(('final', final_results, [budget['effective_total']]))
    for phase, table, steps in phases:
        for i in range(shape[0]):
            for k, step in enumerate(steps):
                for j in range(shape[2]):
                    key = (i, k, j) if phase == 'intermediate' else (i, j)
                    valid = bool(table.validity[key])
                    namespace = 'evaluations' if phase == 'intermediate' else 'final'
                    path = root / 'results' / namespace / f'seed_{i}' / f'w_{j}' / f'step_{step}.json'
                    if path.exists() != valid:
                        raise storage.StorageError('Result validity disagrees with committed record: ' + str(path))
                    record = meta = None
                    if valid:
                        record, meta = storage._load_evaluation_record(path)
                        expected, kind = storage._evaluation_location(root, payload, fingerprint, record.identity, record)
                        if (path != expected or kind != namespace or meta['kind'] != kind
                                or meta['objective_scales'] != science['objective_scales']
                                or not np.array_equal(table.means_raw[key], record.mean_raw)
                                or not np.array_equal(table.means_scaled[key], record.mean_scaled)):
                            raise storage.StorageError('Table means/provenance disagree with committed record')
                        episodes = table.episode_returns if phase == 'intermediate' else table.episode_returns_raw
                        if episodes is not None and not np.array_equal(episodes[key], record.returns_raw):
                            raise storage.StorageError('Table episodes disagree with committed record')
                        if phase == 'intermediate':
                            if key in table.identities and table.identities[key] != record.identity:
                                raise storage.StorageError('Table policy identity disagrees with committed record')
                            metrics = table.episode_metric_tables.get(key)
                        else:
                            if not np.array_equal(table.episode_returns_scaled[key], record.returns_scaled):
                                raise storage.StorageError('Final scaled episodes disagree with committed record')
                            metrics = table.operational_metrics[i][j] if table.operational_metrics else None
                        if metrics is not None and storage._json_value(metrics) != storage._json_value(record.episode_metrics):
                            raise storage.StorageError('Table metrics disagree with committed record')
                        state = attempts[i * shape[2] + j]['state']
                        if (state is not None and step > state['current_step'] and state['resume_attempt_id'] != 'initial'
                                and (meta.get('resume_attempt_id') != state['resume_attempt_id']
                                     or not storage._resumed_bundle_matches(root, record.identity, state['resume_attempt_id'], fingerprint))):
                            raise storage.StorageError('Cannot export a superseded resumed evaluation')
                    base = {'phase': phase, 'experiment_fingerprint': fingerprint,
                        'run_index': i, 'run_seed': science['run_seeds'][i], 'weight_index': j,
                        'weight_tracking': payload['weights']['values'][j][0],
                        'weight_search': payload['weights']['values'][j][1], 'actual_budget': step,
                        'actual_step': step if valid else None,
                        'bundle_id': record.identity.bundle_id if valid else None,
                        'resume_attempt_id': meta.get('resume_attempt_id', 'initial') if valid else None}
                    slots.append((base, record, path, meta))
    return slots, attempts


def _metric_denominator(metric, name, target_count):
    """Use recorded exposure/counts only; never invent estimator sample counts."""
    targets = metric.target_observation_metrics
    if getattr(metric, name) is None:
        return None, metric.unavailable_reasons[name]
    if name == 'mean_coverage':
        return metric.decisions, None
    if name in ('final_coverage', 'search_time_percentage'):
        return (1 if name == 'final_coverage' else metric.elapsed_timesteps), None
    if name in ('mean_known_observation_frequency', 'mean_observation_frequency') and len(targets) == target_count:
        return sum(t.observation_frequency is not None and
                   (name != 'mean_known_observation_frequency' or t.initially_known) for t in targets), None
    if name in ('decisions', 'elapsed_timesteps', 'unique_new_detections'):
        return None, 'Count metric; no averaging denominator'
    return None, 'Within-episode sample count was not recorded in EpisodeMetrics'


def write_summary_tables(results, final_results, experiment_dir):
    """Export validated local records; never load a policy, evaluate or train.

    final_results may be None for an intermediate-only report or an incomplete
    FinalResults. All declared slots appear, with blank cells for missing data.
    Per-policy uncertainty is across episodes; main HV summaries require every
    declared training seed. Operational summaries explicitly use available
    episodes and retain missing counts/reasons and per-target exposure rows.
    Write table_metadata.json last with CSV hashes; consumers must verify them.
    """
    from collections import Counter
    from dataclasses import fields
    import hashlib
    import json
    import numpy as np
    from . import storage
    from .types import EpisodeMetrics
    from .pareto import compute_hv_curve, summarize_hv_across_seeds, hypervolume_2d_max

    root, payload, fingerprint = storage._evaluation_context(experiment_dir)
    slots, attempts = _table_slots(results, final_results, root, payload, fingerprint)
    science = payload['config']
    scalar_metrics = [f.name for f in fields(EpisodeMetrics)
                      if f.name not in ('target_observation_metrics', 'unavailable_reasons')]
    identity_columns = list(slots[0][0])
    episode_columns = identity_columns + ['episode_index', 'episode_seed']
    objectives = ('tracking', 'search')
    summary_columns = [f'{objective}_{stat}_{space}' for space in ('raw', 'scaled')
                       for stat in ('mean', 'sd', 'se') for objective in objectives]
    # Params: base; returns: science['eval_episodes'] or science['final_episodes'].
    def declared(base):
        return science['eval_episodes'] if base['phase'] == 'intermediate' else science['final_episodes']
    # Params: none; returns: iterator.
    def policies():
        for base, record, _, _ in slots:
            row = dict(base, valid=record is not None, declared_episode_count=declared(base),
                       completed_episode_count=0 if record is None else len(record.episode_seeds))
            for space in ('raw', 'scaled'):
                for stat in ('mean', 'sd', 'se'):
                    values = getattr(record, stat + '_' + space) if record else (None, None)
                    row.update({f'{objective}_{stat}_{space}': values[i] for i, objective in enumerate(objectives)})
            yield row
    # Params: none; returns: iterator.
    def episodes():
        for base, record, _, _ in slots:
            if record is None: continue
            for e, seed in enumerate(record.episode_seeds):
                yield dict(base, episode_index=e, episode_seed=seed,
                    **{f'{o}_return_{s}': getattr(record, 'returns_' + s)[e, i]
                       for s in ('raw', 'scaled') for i, o in enumerate(objectives)})
    # Params: none; returns: iterator.
    def episode_metrics():
        for base, record, _, _ in slots:
            if record is None: continue
            for e, metric in enumerate(record.episode_metrics):
                for name in scalar_metrics:
                    count, reason = _metric_denominator(metric, name,
                        science['env']['known_targets'] + science['env']['unknown_targets'])
                    yield dict(base, episode_index=e, episode_seed=record.episode_seeds[e], metric=name,
                        value=getattr(metric, name), unavailable_reason=metric.unavailable_reasons.get(name),
                        within_episode_sample_count=count, sample_count_unavailable_reason=reason,
                        decisions=metric.decisions, elapsed_timesteps=metric.elapsed_timesteps)
    # Params: none; returns: iterator.
    def targets():
        for base, record, _, _ in slots:
            if record is None: continue
            for e, metric in enumerate(record.episode_metrics):
                for target in metric.target_observation_metrics:
                    yield dict(base, episode_index=e, episode_seed=record.episode_seeds[e],
                        **storage._json_value(target),
                        frequency_unavailable_reason=('Never-detected unknown target' if target.observation_frequency is None else None))
    # Params: none; returns: iterator.
    def operational():
        for base, record, _, _ in slots:
            for name in scalar_metrics:
                values = [getattr(m, name) for m in record.episode_metrics if getattr(m, name) is not None] if record else []
                reasons = Counter(m.unavailable_reasons[name] for m in record.episode_metrics if getattr(m, name) is None) if record else Counter({'Missing committed policy evaluation': declared(base)})
                n = len(values)
                sd = float(np.std(values, ddof=1)) if n > 1 else (0. if n else None)
                yield dict(base, metric=name, mean_available_episodes=float(np.mean(values)) if n else None,
                    sd_available_episodes=sd, se_available_episodes=sd / np.sqrt(n) if n else None,
                    available_episode_count=n, missing_episode_count=declared(base) - n,
                    declared_episode_count=declared(base), all_episodes_available=n == declared(base),
                    unavailable_reason_counts=json.dumps(dict(reasons), sort_keys=True))
    # Params: none; returns: iterator.
    def hypervolume():
        phases = [('intermediate', compute_hv_curve(results, science['reference_scaled']),
                   results.validity, results.actual_budgets)]
        if final_results is not None:
            valid = final_results.validity[:, None, :]
            hv = np.full(valid.shape[:2], np.nan, np.float64)
            for i in range(len(valid)):
                if valid[i, 0].all():
                    hv[i, 0] = hypervolume_2d_max(final_results.means_scaled[i], science['reference_scaled'])
            phases.append(('final', hv, valid, np.asarray([payload['budget_plan']['effective_total']])))
        for phase, hv, valid, steps in phases:
            summary = summarize_hv_across_seeds(hv, steps)
            for k, step in enumerate(steps):
                for i, seed in enumerate(science['run_seeds']):
                    yield {'phase': phase, 'row_kind': 'training_seed', 'experiment_fingerprint': fingerprint,
                        'actual_budget': int(step), 'run_index': i, 'run_seed': seed,
                        'complete_policy_count': int(valid[i, k].sum()), 'declared_policy_count': valid.shape[2],
                        'hypervolume': hv[i, k], 'complete_seed_count': int(summary.complete_seed_counts[k]),
                        'declared_seed_count': len(valid), 'all_seeds_complete': bool(summary.completeness[k])}
                yield {'phase': phase, 'row_kind': 'across_training_seeds', 'experiment_fingerprint': fingerprint,
                    'actual_budget': int(step), 'complete_seed_count': int(summary.complete_seed_counts[k]),
                    'declared_seed_count': len(valid), 'all_seeds_complete': bool(summary.completeness[k]),
                    'mean_hypervolume': summary.mean[k], 'sd_hypervolume': summary.sd[k]}

    folder = root / 'reports/tables'
    folder.resolve().relative_to(root)  # Refuse an output parent escaping through a symlink.
    folder.mkdir(parents=True, exist_ok=True)
    tables = [
        ('policy_results.csv', identity_columns + ['valid', 'declared_episode_count', 'completed_episode_count'] + summary_columns, policies()),
        ('episode_returns.csv', episode_columns + [f'{o}_return_{s}' for s in ('raw', 'scaled') for o in objectives], episodes()),
        ('episode_metrics.csv', episode_columns + ['metric', 'value', 'unavailable_reason', 'within_episode_sample_count',
            'sample_count_unavailable_reason', 'decisions', 'elapsed_timesteps'], episode_metrics()),
        ('target_observation_metrics.csv', episode_columns + ['target_id', 'initially_known', 'observation_count',
            'counting_start_timestep', 'eligible_timesteps', 'observation_frequency', 'frequency_unavailable_reason'], targets()),
        ('operational_summary.csv', identity_columns + ['metric', 'mean_available_episodes', 'sd_available_episodes',
            'se_available_episodes', 'available_episode_count', 'missing_episode_count', 'declared_episode_count',
            'all_episodes_available', 'unavailable_reason_counts'], operational()),
        ('hypervolume.csv', ['phase', 'row_kind', 'experiment_fingerprint', 'actual_budget', 'run_index', 'run_seed',
            'complete_policy_count', 'declared_policy_count', 'hypervolume', 'complete_seed_count', 'declared_seed_count',
            'all_seeds_complete', 'mean_hypervolume', 'sd_hypervolume'], hypervolume()),
    ]
    inventory = {}
    for name, columns, rows in tables:
        count = _write_csv(folder / name, columns, rows)
        inventory[name] = {'rows': count, 'columns': columns, 'sha256': hashlib.sha256((folder / name).read_bytes()).hexdigest()}
    provenance = [{'path': path.relative_to(root).as_posix(), 'content_sha256': meta['content_sha256'],
                   'record_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                   'resume_attempt_id': meta.get('resume_attempt_id', 'initial')}
                  for _, record, path, meta in slots if record is not None]
    atomic_write_json(folder / 'table_metadata.json', {
        'schema_version': 1, 'kind': 'derived_summary_tables', 'experiment_fingerprint': fingerprint,
        'normalization': science['normalization'], 'objective_scales': science['objective_scales'],
        'reference_scaled': science['reference_scaled'], 'scale_status': science['scale_status'],
        'reference_status': science['reference_status'], 'reward_order': science['reward_order'],
        'error_target_set': science['error_target_set'], 'gamma': science['ppo']['gamma'],
        'discount_clock': 'decision', 'run_seeds': science['run_seeds'], 'weights': payload['weights'],
        'budget_plan': payload['budget_plan'], 'seed_plans': payload['seed_plans'],
        'final_results_included': final_results is not None, 'resume_attempts': attempts,
        'evaluation_records': provenance, 'tables': inventory,
        'conventions': {
            'null': 'Empty CSV cell; JSON null. Zero is a real observed value.',
            'uncertainty': 'Sample SD (ddof=1) and SE=SD/sqrt(n); singleton SD/SE zero; empty undefined.',
            'operational_summary': 'Equal-episode means over available episodes, with explicit available/missing counts.',
            'hypervolume': 'Scaled reward objectives; per training seed; across-seed mean/SD require all declared seeds.',
            'missing_policy': 'actual_budget is the scheduled slot; blank actual_step means it was not observed.',
            'frequency': 'Observation count / eligible physical timesteps; known from t0, discovered unknown from first detection inclusive.',
            'search_time': 'Undefined for PPO without explicit controller modes; never inferred from rewards/actions.',
            'unrecorded_denominators': 'Positional/covariance target-time sample counts were not retained in EpisodeMetrics; left null.',
        },
    })

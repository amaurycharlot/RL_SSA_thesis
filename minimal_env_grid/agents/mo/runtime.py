"""Process-local execution limits, deliberately absent from scientific plans.

One learner numerical team plus n_envs independent worker teams is the CPU
reservation model. Idle Torch inter-op threads and service threads are reported
but are not additional simultaneously executing numerical teams in eager PPO.
Thread limits are process-wide: callers must not train learners in Python
threads; parallel policies belong in independent processes.
"""
from contextlib import contextmanager, ExitStack
import os


THREAD_ENV = ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
              'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS')


# Params: value, name; returns: value.
def _positive(value, name):
    if type(value) is not int or value < 1:
        raise ValueError(name + ' must be a positive integer')
    return value


def cpu_allocation(cpus_per_task=None, *, controlled=False):
    """Use the tightest explicit/scheduler/affinity bound, never the node size
    in place of a smaller scheduler allocation. Unmanaged defaults stay intact.
    """
    bounds = {}
    if cpus_per_task is not None:
        bounds['cpus_per_task'] = _positive(cpus_per_task, 'cpus_per_task')
    scheduled = os.environ.get('SLURM_CPUS_PER_TASK')
    if scheduled is not None:
        try:
            value = int(scheduled)
        except ValueError:
            raise ValueError('SLURM_CPUS_PER_TASK must be a positive integer') from None
        bounds['SLURM_CPUS_PER_TASK'] = _positive(value, 'SLURM_CPUS_PER_TASK')
    visible = (len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity')
               else os.cpu_count())
    if (controlled or bounds) and visible is not None:
        bounds['visible_cpus'] = _positive(visible, 'visible_cpus')
    return {'allocated_cpus': min(bounds.values()) if bounds else None,
            'allocation_bounds': bounds, 'visible_cpus': visible}


# Params: none; returns: None or list.
def _pools():
    try:
        from threadpoolctl import threadpool_info
    except ImportError:
        return None
    return [{key: row.get(key) for key in
             ('internal_api', 'prefix', 'filepath', 'num_threads')}
            for row in threadpool_info()]


@contextmanager
def execution_resources(config, *, learner_threads=None, cpus_per_task=None):
    """Validate before worker/model creation, apply limits, restore on all exits.

    On managed allocations an explicit learner limit is required. This prevents
    a library imported later from silently expanding an unmanaged thread team.
    Without options or a scheduler allocation the legacy parent limits survive.
    """
    if learner_threads is not None:
        _positive(learner_threads, 'learner_threads')
    allocation = cpu_allocation(cpus_per_task, controlled=learner_threads is not None)
    if learner_threads is not None and allocation['allocated_cpus'] is None:
        raise ValueError('CPU allocation is unknown; supply cpus_per_task (--cpus-per-task)')
    if allocation['allocated_cpus'] is not None and learner_threads is None:
        raise ValueError('CPU allocation requires explicit learner_threads (--learner-threads)')
    workers = config.ppo.n_envs if config.execution.subproc else 0
    worker_threads = config.execution.worker_threads
    demand = None if learner_threads is None else learner_threads + workers * worker_threads
    if allocation['allocated_cpus'] is not None and demand > allocation['allocated_cpus']:
        raise ValueError(f'CPU demand {demand} (learner {learner_threads} + '
                         f'{workers} workers * {worker_threads}) exceeds allocation '
                         f"{allocation['allocated_cpus']}; increase allocation or reduce learner threads")
    import torch
    # Load the same numerical dependencies used by PPO before saving pool state.
    # Otherwise a BLAS library first imported inside the scope would have no
    # previous thread count to restore at exit.
    from stable_baselines3.common.utils import get_device
    previous_threads = torch.get_num_threads()
    previous_env = {key: os.environ.get(key) for key in THREAD_ENV}
    with ExitStack() as owned:
        try:
            if learner_threads is not None:
                try:
                    from threadpoolctl import threadpool_limits
                except ImportError:
                    raise RuntimeError('Learner limits require threadpoolctl; install requirements-mo-runtime.txt') from None
                for key in THREAD_ENV:
                    os.environ[key] = str(learner_threads)
                torch.set_num_threads(learner_threads)
                owned.enter_context(threadpool_limits(limits=learner_threads))
            pools = _pools()
            actual = torch.get_num_threads()
            if learner_threads is not None and (actual > learner_threads or any(
                    row['num_threads'] is not None and row['num_threads'] > learner_threads
                    for row in pools)):
                raise RuntimeError('A learner numerical thread pool exceeds the requested limit')
            yield {**allocation, 'learner_threads_requested': learner_threads,
                   'learner_torch_threads': actual,
                   'learner_torch_interop_threads': torch.get_num_interop_threads(),
                   'learner_numerical_pools': pools,
                   'worker_processes': workers, 'worker_threads': worker_threads,
                   'cpu_demand': demand, 'allocation_checked': demand is not None,
                   'configured_device': config.ppo.device,
                   'resolved_device': str(get_device(config.ppo.device)),
                   'thread_environment': {key: os.environ.get(key) for key in THREAD_ENV}}
        finally:
            # Restore pool limits before Torch; Torch and OpenMP may share a pool.
            owned.close()
            if learner_threads is not None:
                torch.set_num_threads(previous_threads)
                for key, value in previous_env.items():
                    if value is None:
                        os.environ.pop(key, None)
                    else:
                        os.environ[key] = value

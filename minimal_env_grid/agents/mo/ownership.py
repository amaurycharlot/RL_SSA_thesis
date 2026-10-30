"""Exclusive policy-lifetime claims; imports allocate no runtime resources.

An atomic mkdir arbitrates writers on the shared experiment filesystem. The
unique owner filename makes release/recovery unable to unlink a successor's
claim. No timeout, PID guess or scheduler query ever steals an existing claim.
"""
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import socket
from types import MappingProxyType
from typing import Mapping
import uuid

from . import storage


@dataclass(frozen=True)
class PolicyClaim:
    directory: Path
    token: str
    metadata: Mapping


# Params: experiment_dir, config, seed_index, weight_index; returns: (policy, fingerprint).
def _slot(experiment_dir, config, seed_index, weight_index):
    root, payload, fingerprint = storage._evaluation_context(experiment_dir, config)
    if (type(seed_index) is not int or not 0 <= seed_index < len(config.run_seeds)
            or type(weight_index) is not int or not 0 <= weight_index < config.n_weights):
        raise storage.StorageError('Claim indices must identify a declared policy slot')
    if payload['source_hashes'] != storage._source_hashes() or payload['versions'] != storage._versions():
        raise storage.StorageError('Policy ownership requires matching source/package provenance')
    policy = root / 'models' / f'seed_{seed_index}' / f'w_{weight_index}'
    return policy, fingerprint


# Params: none; returns: dict.
def _actor():
    return {'host': socket.gethostname(), 'pid': os.getpid(),
            'slurm_job_id': os.environ.get('SLURM_JOB_ID'),
            'slurm_array_job_id': os.environ.get('SLURM_ARRAY_JOB_ID'),
            'slurm_array_task_id': os.environ.get('SLURM_ARRAY_TASK_ID'),
            'slurm_restart_count': os.environ.get('SLURM_RESTART_COUNT')}


def _release(directory, token):
    # Only the process that successfully removes this unique owner file may
    # remove the directory. A delayed former owner cannot touch a replacement.
    try:
        (directory / (token + '.json')).unlink()
    except FileNotFoundError as error:
        raise storage.StorageError('Policy claim owner changed; replacement claim was not removed') from error
    try:
        directory.rmdir()
    except OSError as error:
        raise storage.StorageError('Policy claim is not empty; explicit recovery is required') from error


@contextmanager
def _claim_directory(directory, metadata, label):
    """Publish an exclusive, token-owned claim; preserve it on hard exit."""
    token = uuid.uuid4().hex
    metadata = dict(metadata, token=token)
    encoded = storage._json_bytes(metadata)
    try:
        directory.mkdir()
    except FileExistsError as error:
        raise storage.StorageError(
            f'{label} already claimed: {directory}; inspect its owner and recover explicitly only after it stops') from error
    owner = directory / (token + '.json')
    try:
        with owner.open('xb') as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        if owner.exists():
            owner.unlink()
        directory.rmdir()
        raise
    try:
        yield PolicyClaim(directory, token, MappingProxyType(metadata))
    finally:
        _release(directory, token)


@contextmanager
def claim_policy_slot(experiment_dir, config, seed_index, weight_index):
    """Own one policy through cleanup/completion, refusing an active finalizer.

    The checks before and after atomic claim publication close the startup race:
    a finalizer either observes this claim, or this worker refuses to yield while
    finalization owns the experiment. No age-based stale takeover is performed.
    """
    policy, fingerprint = _slot(experiment_dir, config, seed_index, weight_index)
    finalizer = policy.parents[2] / 'FINALIZER.LOCK'
    if finalizer.exists():
        raise storage.StorageError('Experiment finalizer is active/interrupted; policy claims are blocked')
    metadata = dict(_actor(), schema_version=1,
        created_at=datetime.now(timezone.utc).isoformat(),
        experiment_fingerprint=fingerprint, seed_index=seed_index,
        run_seed=config.run_seeds[seed_index], weight_index=weight_index)
    policy.mkdir(parents=True, exist_ok=True)
    with _claim_directory(policy / 'OWNER.LOCK', metadata, 'Policy slot') as claim:
        if finalizer.exists():
            raise storage.StorageError('Experiment finalizer started during policy claim publication')
        yield claim


@contextmanager
def claim_finalizer(experiment_dir, config):
    """Exclude other finalizers and prevent newly starting policy owners.

    The caller must inspect existing policy/storage claims after this yields,
    before touching shared artifacts. Hard exit requires explicit token recovery.
    """
    root, payload, fingerprint = storage._evaluation_context(experiment_dir, config)
    if payload['source_hashes'] != storage._source_hashes() or payload['versions'] != storage._versions():
        raise storage.StorageError('Finalization requires matching source/package provenance')
    metadata = dict(_actor(), schema_version=1, kind='finalizer',
        created_at=datetime.now(timezone.utc).isoformat(), experiment_fingerprint=fingerprint)
    with _claim_directory(root / 'FINALIZER.LOCK', metadata, 'Experiment finalizer') as claim:
        yield claim



def recover_policy_claim(experiment_dir, config, seed_index, weight_index, *,
                         expected_token, owner_stopped=False, reason):
    """Explicitly release an inspected, stopped owner's stale claim.

    The operator must first stop/verify the recorded process AND scheduler job,
    including on remote hosts, then supply its exact token and a reason. No age
    or local PID test proves a remote owner stopped. Current-process claims are
    always refused. An empty directory left by death before metadata publication
    has no owner token: stop all writers/submitters before manually removing
    that empty directory. This API refuses empty/tokenless recovery, avoiding
    races with a new owner. Partial metadata can be recovered by filename token.
    An authorization record is persisted before changing the stale claim.
    """
    if owner_stopped is not True:
        raise storage.StorageError('Explicit recovery requires owner_stopped=True after verifying process/job termination')
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError('Recovery requires a nonempty reason')
    if (not isinstance(expected_token, str)
            or re.fullmatch(r'[0-9a-f]{32}', expected_token) is None):
        raise ValueError('expected_token must be the exact 32-character owner token; tokenless recovery requires offline inspection')
    policy, fingerprint = _slot(experiment_dir, config, seed_index, weight_index)
    directory = policy / 'OWNER.LOCK'
    if not directory.is_dir():
        raise storage.StorageError('No policy claim to recover')
    owner = directory / (expected_token + '.json')
    try:
        raw = owner.read_bytes()
    except FileNotFoundError as error:
        raise storage.StorageError('Expected owner token does not match the current claim') from error
    metadata = None
    try:
        metadata = json.loads(raw)
    except (ValueError, UnicodeError):
        pass  # Incomplete metadata is recoverable only by explicit token.
    if isinstance(metadata, dict):
        if metadata.get('host') == socket.gethostname() and metadata.get('pid') == os.getpid():
            raise storage.StorageError('Cannot recover a claim owned by the current process')
    authorization = dict(_actor(), schema_version=1, kind='stale_policy_claim_recovery_authorization',
        authorized_at=datetime.now(timezone.utc).isoformat(), reason=reason.strip(),
        owner_stopped=True, expected_token=expected_token, owner_metadata=metadata,
        owner_sha256=hashlib.sha256(raw).hexdigest(), experiment_fingerprint=fingerprint,
        seed_index=seed_index, weight_index=weight_index)
    audit = policy / 'claim_recovery'
    audit.mkdir(exist_ok=True)
    path = audit / (uuid.uuid4().hex + '.json')
    storage.atomic_write_json(path, authorization)
    _release(directory, expected_token)
    return path


def recover_finalizer_claim(experiment_dir, config, *,
                         expected_token, owner_stopped=False, reason):
    """Explicitly release an inspected, stopped owner's stale finalizer claim.

    The operator must first stop/verify the recorded process AND scheduler job,
    including on remote hosts, then supply its exact token and a reason. No age
    or local PID test proves a remote owner stopped. Current-process claims are
    always refused. An empty directory left by death before metadata publication
    has no owner token: stop all writers/submitters before manually removing
    that empty directory. This API refuses empty/tokenless recovery, avoiding
    races with a new owner. Partial metadata can be recovered by filename token.
    An authorization record is persisted before changing the stale finalizer claim.
    """
    if owner_stopped is not True:
        raise storage.StorageError('Explicit recovery requires owner_stopped=True after verifying process/job termination')
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError('Recovery requires a nonempty reason')
    if (not isinstance(expected_token, str)
            or re.fullmatch(r'[0-9a-f]{32}', expected_token) is None):
        raise ValueError('expected_token must be the exact 32-character owner token; tokenless recovery requires offline inspection')
    policy, payload, fingerprint = storage._evaluation_context(experiment_dir, config)
    if payload['source_hashes'] != storage._source_hashes() or payload['versions'] != storage._versions():
        raise storage.StorageError('Finalizer recovery requires matching source/package provenance')
    directory = policy / 'FINALIZER.LOCK'
    if not directory.is_dir():
        raise storage.StorageError('No finalizer claim to recover')
    owner = directory / (expected_token + '.json')
    try:
        raw = owner.read_bytes()
    except FileNotFoundError as error:
        raise storage.StorageError('Expected owner token does not match the current claim') from error
    metadata = None
    try:
        metadata = json.loads(raw)
    except (ValueError, UnicodeError):
        pass  # Incomplete metadata is recoverable only by explicit token.
    if isinstance(metadata, dict):
        if metadata.get('host') == socket.gethostname() and metadata.get('pid') == os.getpid():
            raise storage.StorageError('Cannot recover a claim owned by the current process')
    authorization = dict(_actor(), schema_version=1, kind='stale_finalizer_claim_recovery_authorization',
        authorized_at=datetime.now(timezone.utc).isoformat(), reason=reason.strip(),
        owner_stopped=True, expected_token=expected_token, owner_metadata=metadata,
        owner_sha256=hashlib.sha256(raw).hexdigest(), experiment_fingerprint=fingerprint)
    audit = policy / 'finalization_recovery'
    audit.mkdir(exist_ok=True)
    path = audit / (uuid.uuid4().hex + '.json')
    storage.atomic_write_json(path, authorization)
    _release(directory, expected_token)
    return path

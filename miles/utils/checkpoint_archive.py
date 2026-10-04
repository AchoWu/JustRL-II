"""Unrotated copies of dist checkpoints, for pairing a non-final actor with its critic.

Megatron rotates every dist checkpoint but the latest, and the critic has no HF export
(``_should_save_hf`` in ``actor.py`` returns False for ``role != "actor"``). So at the
end of a run the only critic left on disk is the last one — and when the best actor is
an earlier step, there is no critic to go with it.

``--save-retain-interval`` cannot fill the gap. A save fires when ``rollout_id + 1``
reaches a multiple of ``save_interval`` (``misc.should_run_periodic_action`` uses
``step = rollout_id + 1``) while the directory is named for ``rollout_id``, so the
iterations on disk are ``k*save_interval - 1``. Megatron keeps ``it`` only when
``it % save_retain_interval == 0``, and ``save_retain_interval`` is itself asserted to
be a multiple of ``save_interval`` — so retention needs ``k*S - 1 ≡ 0 (mod m*S)``,
which has no solution for ``S > 1``.

Kept free of torch/megatron/ray imports so it is testable without the training stack:
the collective half (the barrier, the single-writer choice) stays in ``actor.py``.
"""

import logging
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)

# `.metadata` is a dotfile, and together with the `__*.distcp` shards it IS the
# checkpoint as far as torch.distributed.checkpoint is concerned. Note `ll` hides it,
# so a valid checkpoint can look like bare shards.
METADATA_NAME = ".metadata"

__all__ = ["archive_dir_for", "is_archive_point", "archive_checkpoint"]


def archive_dir_for(save_dir: str | Path) -> Path:
    """`<save_dir>_archive` — a sibling, so Megatron's rotation cannot reclaim it."""
    return Path(f"{str(save_dir).rstrip('/')}_archive")


def is_archive_point(rollout_id: int, interval: int | None, num_rollout: int | None) -> bool:
    """Whether this rollout should be archived.

    A pure function of ``rollout_id`` and static args on purpose: the caller takes a
    collective barrier when this returns True, so every rank has to reach the same
    answer or the run deadlocks.
    """
    if not interval:
        return False
    if num_rollout is not None and rollout_id == num_rollout - 1:
        return True
    return (rollout_id + 1) % interval == 0


def archive_checkpoint(save_dir: str | Path, rollout_id: int) -> Path | None:
    """Copy ``<save_dir>/iter_xxx`` to ``<save_dir>_archive/iter_xxx``.

    Returns the archive path, or None when nothing was copied (already archived, or the
    source is not a complete checkpoint). Copies rather than hardlinks or moves: the
    source is about to be rotated away, and the shards are plain files opened by name.

    Never raises on an I/O failure — losing an archive copy must not kill a run that is
    otherwise healthy. The caller is mid-``save_model`` with the process groups
    reloaded; an exception here would take the whole rollout down.
    """
    src = Path(save_dir) / f"iter_{rollout_id:07d}"
    if not (src / METADATA_NAME).is_file():
        logger.warning("[archive] %s has no %s, skipping archive copy", src, METADATA_NAME)
        return None

    dst = archive_dir_for(save_dir) / src.name
    if (dst / METADATA_NAME).is_file():
        logger.info("[archive] %s already archived", dst)
        return dst

    # Copy into `.tmp` and rename, so an interrupted copy never leaves behind a
    # directory that looks complete.
    tmp = dst.with_name(f"{dst.name}.tmp")
    try:
        shutil.rmtree(tmp, ignore_errors=True)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(src, tmp)
        if dst.is_dir():
            shutil.rmtree(dst)
        tmp.rename(dst)
    except OSError as e:
        logger.warning("[archive] failed to archive %s: %s", src, e)
        shutil.rmtree(tmp, ignore_errors=True)
        return None

    logger.info("[archive] %s -> %s", src, dst)
    return dst

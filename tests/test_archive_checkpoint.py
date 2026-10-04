"""`--archive-checkpoint-interval`: unrotated copies, so a non-final actor keeps a critic.

Megatron rotates all but the latest dist checkpoint, and the critic has no HF export
(`_should_save_hf` in `actor.py` returns False for `role != "actor"`). So by the end of
a run the only surviving critic is the last one — and when the best actor is an earlier
step (the 16k run's eval peaks at rollout 399), there is no critic to pair with it.

`--save-retain-interval` looks like the built-in answer and provably cannot do it:
`test_save_retain_interval_can_never_retain_anything` is the reason this feature
exists, so it is pinned first.

`miles/utils/checkpoint_archive.py` is deliberately free of torch/megatron/ray imports
so these run on a plain CPU box; the collective half (barrier, single writer) stays in
`actor.py` and is covered by the pure-function contract `is_archive_point` must satisfy.
"""

from pathlib import Path

import pytest

from miles.utils.checkpoint_archive import archive_checkpoint, archive_dir_for, is_archive_point


def saved_iterations(num_rollout: int, save_interval: int) -> list[int]:
    """Iterations that land on disk, mirroring should_run_periodic_action + train.py.

    `misc.should_run_periodic_action` sets `step = rollout_id + 1` and fires on
    `step % interval == 0`; `train.py:save` then names the directory for `rollout_id`.
    """
    return [r for r in range(num_rollout) if (r + 1) % save_interval == 0 or r == num_rollout - 1]


@pytest.mark.parametrize("save_interval", [25, 50])
@pytest.mark.parametrize("multiple", [1, 2, 4, 10])
def test_save_retain_interval_can_never_retain_anything(save_interval, multiple):
    """Why this feature exists: the built-in knob retains nothing, for ANY legal value.

    Megatron keeps `it` when `it % save_retain_interval == 0`, and train.sh asserts
    `save_retain_interval % save_interval == 0`. On-disk iterations are
    `k*save_interval - 1`, so retention needs `k*S - 1 ≡ 0 (mod m*S)` — unsolvable for
    `S > 1`.
    """
    retain = save_interval * multiple
    on_disk = saved_iterations(500, save_interval)

    assert [it for it in on_disk if it % retain == 0] == []


def test_saved_iterations_are_off_by_one_from_the_interval():
    """Where the off-by-one comes from: save fires at step k*S, dir named k*S - 1."""
    assert saved_iterations(500, 25)[:4] == [24, 49, 74, 99]
    assert saved_iterations(500, 25)[-1] == 499


# --- is_archive_point: must be a pure function of rollout_id and static args, because
# --- actor.py takes a collective barrier when it returns True.


@pytest.mark.parametrize(
    "rollout_id,expected",
    [(24, False), (49, True), (74, False), (99, True), (499, True)],
)
def test_is_archive_point_follows_the_plus_one_convention(rollout_id, expected):
    assert is_archive_point(rollout_id, 50, 500) is expected


def test_final_rollout_is_always_an_archive_point():
    """499 is not a multiple of 50 under the +1 convention, but it is the last step."""
    assert is_archive_point(499, 50, 500) is True
    assert is_archive_point(499, 200, 500) is True


def test_is_archive_point_disabled_without_an_interval():
    assert is_archive_point(49, None, 500) is False
    assert is_archive_point(49, 0, 500) is False


def test_is_archive_point_is_deterministic():
    """Same inputs, same answer — if it could vary per rank, the barrier would deadlock."""
    assert all(is_archive_point(49, 50, 500) for _ in range(5))


def test_archive_dir_is_a_sibling_not_a_child(tmp_path):
    """It must sit outside the save dir, or Megatron's rotation reclaims it."""
    save_dir = tmp_path / "run"

    archive = archive_dir_for(save_dir)

    assert archive.name == "run_archive"
    assert archive.parent == save_dir.parent
    assert not str(archive).startswith(str(save_dir) + "/")
    # A trailing slash must not produce "run/_archive".
    assert archive_dir_for(f"{save_dir}/") == archive


def _make_ckpt(save_dir: Path, rollout_id: int, *, with_metadata: bool = True) -> Path:
    src = save_dir / f"iter_{rollout_id:07d}"
    src.mkdir(parents=True)
    (src / "__0_0.distcp").write_text("shard0")
    (src / "__1_0.distcp").write_text("shard1")
    if with_metadata:
        (src / ".metadata").write_text("meta")
    return src


def test_archive_copies_shards_and_the_metadata_dotfile(tmp_path):
    save_dir = tmp_path / "run"
    _make_ckpt(save_dir, 49)

    dst = archive_checkpoint(save_dir, 49)

    assert dst == archive_dir_for(save_dir) / "iter_0000049"
    # The dotfile is easy to miss (ll hides it) and the checkpoint is unloadable
    # without it, so assert it explicitly alongside the shards.
    assert (dst / ".metadata").read_text() == "meta"
    assert (dst / "__0_0.distcp").read_text() == "shard0"
    assert (dst / "__1_0.distcp").read_text() == "shard1"


def test_archive_survives_rotation_of_the_source(tmp_path):
    """The whole point: the source is deleted right after, the archive must remain."""
    import shutil

    save_dir = tmp_path / "run"
    src = _make_ckpt(save_dir, 49)
    dst = archive_checkpoint(save_dir, 49)

    shutil.rmtree(src)  # what Megatron's rotation does

    assert (dst / ".metadata").is_file()
    assert (dst / "__0_0.distcp").read_text() == "shard0"


def test_incomplete_checkpoint_is_not_archived(tmp_path, caplog):
    """Without .metadata the tree is not loadable; warn, never raise."""
    save_dir = tmp_path / "run"
    _make_ckpt(save_dir, 49, with_metadata=False)

    assert archive_checkpoint(save_dir, 49) is None
    assert not (archive_dir_for(save_dir) / "iter_0000049").exists()
    assert "no .metadata" in caplog.text


def test_missing_source_is_not_archived(tmp_path, caplog):
    assert archive_checkpoint(tmp_path / "run", 49) is None
    assert "no .metadata" in caplog.text


def test_rerun_does_not_recopy(tmp_path, caplog):
    import logging

    save_dir = tmp_path / "run"
    _make_ckpt(save_dir, 49)
    archive_checkpoint(save_dir, 49)
    (archive_dir_for(save_dir) / "iter_0000049" / "marker").write_text("x")
    caplog.clear()

    with caplog.at_level(logging.INFO, logger="miles.utils.checkpoint_archive"):
        archive_checkpoint(save_dir, 49)

    assert "already archived" in caplog.text
    # Untouched, not re-copied.
    assert (archive_dir_for(save_dir) / "iter_0000049" / "marker").is_file()


def test_stale_tmp_dir_is_replaced(tmp_path):
    """An interrupted copy leaves iter_xxx.tmp; the next attempt must not trip on it."""
    save_dir = tmp_path / "run"
    _make_ckpt(save_dir, 49)
    stale = archive_dir_for(save_dir) / "iter_0000049.tmp"
    stale.mkdir(parents=True)
    (stale / "junk").write_text("partial")

    dst = archive_checkpoint(save_dir, 49)

    assert (dst / ".metadata").is_file()
    assert not (dst / "junk").exists()
    assert not stale.exists()


def test_partial_archive_without_metadata_is_overwritten(tmp_path):
    """A dir that exists but lacks .metadata is a failed copy, not a done one."""
    save_dir = tmp_path / "run"
    _make_ckpt(save_dir, 49)
    partial = archive_dir_for(save_dir) / "iter_0000049"
    partial.mkdir(parents=True)
    (partial / "junk").write_text("partial")

    dst = archive_checkpoint(save_dir, 49)

    assert (dst / ".metadata").is_file()
    assert not (dst / "junk").exists()


def test_actor_and_critic_archive_independently(tmp_path):
    """Both roles call this with their own args.save, so one interval keeps the pair
    aligned — which is the point: a best-step actor needs the matching critic."""
    actor_dir, critic_dir = tmp_path / "run", tmp_path / "run_critic"
    _make_ckpt(actor_dir, 49)
    _make_ckpt(critic_dir, 49)

    archive_checkpoint(actor_dir, 49)
    archive_checkpoint(critic_dir, 49)

    assert (tmp_path / "run_archive" / "iter_0000049" / ".metadata").is_file()
    assert (tmp_path / "run_critic_archive" / "iter_0000049" / ".metadata").is_file()
    # The critic archive must not land inside the actor's.
    assert not (tmp_path / "run_archive" / "run_critic_archive").exists()


def test_archive_point_and_copy_agree_on_the_directory_name(tmp_path):
    """Guards the integration seam: actor.py tests the rollout_id, then copies
    iter_<rollout_id>. A mismatch in zero-padding would archive nothing."""
    save_dir = tmp_path / "run"
    archived = []
    for rollout_id in range(100):
        if is_archive_point(rollout_id, 50, 100):
            _make_ckpt(save_dir, rollout_id)
            dst = archive_checkpoint(save_dir, rollout_id)
            assert dst is not None
            archived.append(dst.name)

    assert archived == ["iter_0000049", "iter_0000099"]

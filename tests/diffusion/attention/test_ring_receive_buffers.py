# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from collections.abc import Callable
from contextlib import nullcontext

import pytest
import torch

from tests.helpers.mark import hardware_test
from vllm_omni.diffusion.attention.backends import ring_flash_attn
from vllm_omni.diffusion.attention.backends.ring import ring_utils
from vllm_omni.diffusion.attention.backends.ring.ring_selector import AttnType

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion]
requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.hip is not None, reason="requires NVIDIA CUDA"
)


class _FakeComm:
    """Simulate receives while checking the actual native loop's buffer ownership."""

    def __init__(self, rank: int, shards: tuple[list[torch.Tensor], list[torch.Tensor]]):
        self.rank = rank
        self.world_size = len(shards[0])
        self.shards = shards
        self.allocations: list[torch.Tensor] = []
        self.receives: list[torch.Tensor] = []
        self.provided: list[bool] = []
        self.pending: list[tuple[torch.Tensor, torch.Tensor]] = []
        self.commits = 0
        self.waits = 0
        self.merge_flags: list[bool] = []

    def send_recv(self, tensor: torch.Tensor, recv_tensor: torch.Tensor | None = None) -> torch.Tensor:
        channel = len(self.pending)
        assert channel < 2
        expected = self.shards[channel][(self.rank - self.commits) % self.world_size]
        torch.testing.assert_close(tensor, expected, rtol=0, atol=0)
        self.provided.append(recv_tensor is not None)
        if recv_tensor is None:
            recv_tensor = torch.empty_like(tensor)
            self.allocations.append(recv_tensor)
        assert tensor.data_ptr() != recv_tensor.data_ptr()
        assert all(
            recv_tensor.data_ptr() != shard.data_ptr() for channel_shards in self.shards for shard in channel_shards
        )
        self.receives.append(recv_tensor)
        self.pending.append((tensor, recv_tensor))
        return recv_tensor

    def commit(self) -> None:
        assert len(self.pending) == 2 and self.waits == self.commits
        block_rank = (self.rank - self.commits - 1) % self.world_size
        for channel, (_, receiver) in enumerate(self.pending):
            receiver.copy_(self.shards[channel][block_rank])
        self.commits += 1

    def wait(self) -> None:
        assert self.commits == self.waits + 1
        self.pending.clear()
        self.waits += 1


def _install_fake_ring(
    monkeypatch: pytest.MonkeyPatch, device: str, world: int, rank: int, offset: int = 0
) -> tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor], list[_FakeComm], list[int]]:
    shape = (1, 2, 2, 8)
    keys = [torch.full(shape, index + offset, dtype=torch.bfloat16, device=device) for index in range(world)]
    values = [torch.full(shape, 2 * index + 1 + offset, dtype=torch.bfloat16, device=device) for index in range(world)]
    instances: list[_FakeComm] = []
    attended: list[int] = []

    def make_comm(_group: object) -> _FakeComm:
        comm = _FakeComm(rank, (keys, values))
        instances.append(comm)
        return comm

    def attention(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        softmax_scale: float,
        causal: bool,
        **_unused: object,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        attended.append(int(k[0, 0, 0, 0].item()) - offset)
        scores = torch.einsum("bqhd,bkhd->bhqk", q.float(), k.float()) * softmax_scale
        if causal:
            mask = torch.ones(q.shape[1], k.shape[1], device=q.device, dtype=torch.bool).triu(1)
            scores = scores.masked_fill(mask, -torch.inf)
        out = torch.einsum("bhqk,bkhd->bqhd", scores.softmax(-1), v.float())
        return out, scores.logsumexp(-1)

    def merge(out, lse, block_out, block_lse, *, lse_layout, use_fused_merge=False):
        instances[-1].merge_flags.append(use_fused_merge)
        return ring_utils.update_out_and_lse(
            out, lse, block_out, block_lse, lse_layout=lse_layout, use_fused_merge=use_fused_merge
        )

    monkeypatch.setattr(ring_flash_attn, "RingComm", make_comm)
    monkeypatch.setattr(ring_flash_attn, "update_out_and_lse", merge)
    monkeypatch.setattr(ring_flash_attn, "select_flash_attn_impl", lambda *_args, **_kwargs: attention)
    monkeypatch.setattr(torch.distributed, "get_backend", lambda _group: "nccl")
    q = torch.zeros(shape, dtype=torch.bfloat16, device=device)
    return (q, keys[rank], values[rank]), instances, attended


def _assert_receives(comm: _FakeComm, *, reuse: bool) -> None:
    assert comm.commits == comm.waits == comm.world_size - 1
    exchanges = 2 * (comm.world_size - 1)
    assert len(comm.receives) == exchanges
    assert len(comm.allocations) == (4 if reuse else exchanges)
    assert comm.provided == ([False] * 4 + [True] * (exchanges - 4) if reuse else [False] * exchanges)
    if reuse:
        assert len({tensor.data_ptr() for tensor in comm.receives[:4]}) == 4
        for index, tensor in enumerate(comm.receives):
            assert tensor is comm.receives[index % 4]


@hardware_test(res={"cuda": "L4"}, num_cards=1)
@requires_cuda
@pytest.mark.parametrize(
    "world,rank,causal,backend,hip,reuse,fused",
    [
        (1, 0, False, AttnType.FA3, False, False, False),
        (2, 1, False, AttnType.FA3, False, False, True),
        (3, 2, False, AttnType.FA3, False, False, True),
        (4, 3, False, AttnType.FA3, False, True, True),
        (5, 4, False, AttnType.FA3, False, True, True),
        (8, 7, False, AttnType.FA3, False, True, True),
        (5, 0, True, AttnType.FA3, False, True, True),
        (5, 2, True, AttnType.FA3, False, True, True),
        (5, 4, False, AttnType.FA, False, True, True),
        (4, 3, False, AttnType.FA4, False, True, False),
        (5, 4, False, AttnType.FA4, False, True, False),
        (8, 7, False, AttnType.FA4, False, True, False),
        (5, 4, False, AttnType.TORCH, False, True, False),
        (4, 3, False, AttnType.FA3, True, True, False),
        (5, 4, False, AttnType.FA3, True, True, False),
        (8, 7, False, AttnType.FA3, True, True, False),
        (1, 0, False, AttnType.AITER, True, False, False),
        (2, 1, False, AttnType.AITER, True, False, False),
        (3, 2, False, AttnType.AITER, True, False, False),
        (4, 3, False, AttnType.AITER, True, True, False),
        (5, 4, False, AttnType.AITER, True, True, False),
        (8, 7, False, AttnType.AITER, True, True, False),
        (5, 4, False, AttnType.FLASHINFER, False, False, False),
        (5, 4, False, AttnType.SPARSE_SAGE, False, False, False),
    ],
)
@torch.inference_mode()
def test_native_ring_receive_slots(monkeypatch, world, rank, causal, backend, hip, reuse, fused):
    # One GPU suffices for ownership checks. Real NCCL stream ordering is a
    # separate distributed qualification; this fake deliberately has no NCCL.
    # Mocking HIP here checks dispatch only, not execution on ROCm hardware.
    if hip:
        monkeypatch.setattr(torch.version, "hip", "test-rocm")
    live_allocations: list[torch.Tensor] = []
    for offset in (0, 20):
        inputs, instances, attended = _install_fake_ring(monkeypatch, "cuda", world, rank, offset)
        snapshots = [tensor.clone() for tensor in inputs]
        expected_visits = [(rank - step) % world for step in range(rank + 1 if causal else world)]
        options = dict(softmax_scale=8**-0.5, causal=causal, attn_type=backend)

        # The same native function's grad-enabled route retains fresh receives.
        with torch.enable_grad():
            reference = ring_flash_attn.ring_flash_attn_forward(None, *inputs, **options)
        actual = ring_flash_attn.ring_flash_attn_forward(None, *inputs, **options)
        for tensor, expected in zip(actual, reference):
            torch.testing.assert_close(tensor, expected, rtol=2e-6, atol=2e-6)
        for tensor, snapshot in zip(inputs, snapshots):
            torch.testing.assert_close(tensor, snapshot, rtol=0, atol=0)
        assert attended == expected_visits * 2
        merge_count = 0 if backend == AttnType.SPARSE_SAGE else len(expected_visits)
        assert instances[0].merge_flags == [False] * merge_count
        assert instances[1].merge_flags == [fused] * merge_count
        _assert_receives(instances[0], reuse=False)
        _assert_receives(instances[1], reuse=reuse)
        for comm in instances:
            prior_pointers = {tensor.data_ptr() for tensor in live_allocations}
            assert not ({tensor.data_ptr() for tensor in comm.allocations} & prior_pointers)
            live_allocations.extend(comm.allocations)


@hardware_test(res={"cuda": "L4"}, num_cards=1)
@requires_cuda
@pytest.mark.parametrize("fallback", ["grad", "compile", "capture", "backend", "device"])
@pytest.mark.parametrize("hip", [False, True])
@torch.inference_mode()
def test_native_ring_receive_fallbacks(monkeypatch, fallback, hip):
    if hip:
        monkeypatch.setattr(torch.version, "hip", "test-rocm")
    inputs, instances, _ = _install_fake_ring(monkeypatch, "cuda", world=5, rank=4)
    device_index = inputs[0].device.index
    assert device_index is not None
    overrides: dict[str, tuple[object, str, Callable[[], object]]] = {
        "compile": (torch.compiler, "is_compiling", lambda: True),
        "capture": (torch.cuda, "is_current_stream_capturing", lambda: True),
        "device": (torch.accelerator, "current_device_index", lambda: device_index + 1),
    }
    if fallback in overrides:
        target, name, value = overrides[fallback]
        monkeypatch.setattr(target, name, value)
    elif fallback == "backend":
        monkeypatch.setattr(torch.distributed, "get_backend", lambda _group: "gloo")
    with torch.enable_grad() if fallback == "grad" else nullcontext():
        output, lse = ring_flash_attn.ring_flash_attn_forward(
            None, *inputs, softmax_scale=8**-0.5, causal=False, attn_type=AttnType.FA3
        )
    assert torch.isfinite(output).all() and torch.isfinite(lse).all()
    assert instances[0].merge_flags == [False] * 5
    _assert_receives(instances[0], reuse=False)


@pytest.mark.cpu
@torch.inference_mode()
def test_native_ring_cpu_keeps_fresh_receives(monkeypatch):
    inputs, instances, _ = _install_fake_ring(monkeypatch, "cpu", world=5, rank=4)

    def unexpected_cuda_query() -> bool:
        raise AssertionError("CPU Ring must not query CUDA capture state")

    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", unexpected_cuda_query)
    output, lse = ring_flash_attn.ring_flash_attn_forward(
        None, *inputs, softmax_scale=8**-0.5, causal=False, attn_type=AttnType.FA3
    )
    assert torch.isfinite(output).all() and torch.isfinite(lse).all()
    assert instances[0].merge_flags == [False] * 5
    _assert_receives(instances[0], reuse=False)

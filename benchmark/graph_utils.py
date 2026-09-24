"""Scoped graph validation/timing for the AllGather-KV experiment."""

import torch
import torch.distributed as dist


def measure_graph(fn, inputs, measure, iterations, repeats, warmup):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            result = fn()
            del result
    stream.synchronize()
    dist.barrier()
    torch.cuda.empty_cache()
    initial_allocated = torch.cuda.memory_allocated()
    initial_reserved = torch.cuda.memory_reserved()
    torch.cuda.reset_peak_memory_stats()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        graph_output = fn()
    torch.cuda.synchronize()
    memory = torch.tensor(
        [
            torch.cuda.max_memory_allocated() - initial_allocated,
            torch.cuda.memory_reserved() - initial_reserved,
        ],
        dtype=torch.int64,
        device="cuda",
    )
    dist.all_reduce(memory, op=dist.ReduceOp.MAX)
    memory = memory.tolist()

    # A changed-input replay must recompute, not return the captured warmup.
    original = [tensor.clone() for tensor in inputs]
    for tensor in inputs:
        tensor.mul_(0.9)
    expected = fn()
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(graph_output, expected, rtol=0.005, atol=0.005)
    error = (graph_output.float() - expected.float()).abs().max()
    dist.all_reduce(error, op=dist.ReduceOp.MAX)
    max_error = error.item()
    for tensor, saved in zip(inputs, original):
        tensor.copy_(saved)
    del tensor, saved, original, expected, error
    graph.replay()
    torch.cuda.synchronize()
    samples = [measure(graph.replay, iterations, warmup) for _ in range(repeats)]
    graph.reset()
    del graph, graph_output
    torch.cuda.synchronize()
    return {
        "samples": samples,
        "capture_peak_allocated_increment_bytes": memory[0],
        "capture_reserved_increment_bytes": memory[1],
        "changed_input_max_abs_error": max_error,
    }

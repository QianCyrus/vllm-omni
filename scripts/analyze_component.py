"""Summarize complete four-rank VAE batch results without dropping cases."""

import argparse
import json
from pathlib import Path
import statistics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    root = args.run
    assert (root / "exit_code.txt").read_text().strip() == "0"
    ranks = []
    checks = []
    for rank in range(4):
        assert json.loads((root / f"done-rank{rank}.json").read_text())["cleanup_complete"]
        ranks.append(json.loads((root / f"timing-rank{rank}.json").read_text()))
        checks.extend(json.loads((root / f"correctness-rank{rank}.json").read_text()))
    assert all(len(rows) == 16 for rows in ranks)
    output = []
    for batch in (1, 2, 4, 5):
        arms = {}
        for arm in ("base", "batch"):
            rows = [row for row in ranks[0] if row["batch"] == batch and row["arm"] == arm]
            samples = [max(times) for row in rows for times in row["per_rank_wall_ms"]]
            arms[arm] = {
                "mean_ms": statistics.mean(samples),
                "sample_std_ms": statistics.stdev(samples),
                "round_means_ms": [row["mean_slowest_rank_ms"] for row in rows],
                "samples_ms": samples,
                "max_rank_peak_allocated_mib": max(
                    row["peak_allocated_bytes_local"] / 2**20
                    for rank_rows in ranks for row in rank_rows
                    if row["batch"] == batch and row["arm"] == arm
                ),
            }
        comparisons = [c for c in checks if c["batch"] == batch]
        assert all(c["finite"] and c["dtype_matches"] and c["shape_matches"] for c in comparisons)
        reduction = (1 - arms["batch"]["mean_ms"] / arms["base"]["mean_ms"]) * 100
        output.append({"batch_size": batch, **arms, "latency_reduction_pct": reduction,
                       "worst_max_abs": max(c["max_abs"] for c in comparisons),
                       "worst_mae": max(c["mae"] for c in comparisons),
                       "worst_rmse": max(c["rmse"] for c in comparisons)})
    result = {"scope": "native VAE component including communication; synthetic latents, not full-model HTTP",
              "rank_count": 4, "warmups_per_round": 2, "repeats_per_round": 8, "rounds_per_arm": 2,
              "results": output}
    (root / "component-summary.json").write_text(json.dumps(result, indent=2) + "\n")
    for row in output:
        print(f"B{row['batch_size']}: {row['base']['mean_ms']:.3f} -> {row['batch']['mean_ms']:.3f} ms; "
              f"reduction {row['latency_reduction_pct']:.2f}%; max_abs {row['worst_max_abs']}")


if __name__ == "__main__":
    main()

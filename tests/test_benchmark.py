import pytest
from click.testing import CliRunner

from log_psplines.benchmark.cli import main as benchmark_cli
from log_psplines.benchmark.runtime_benchmark import RuntimeBenchmark
import os


def test_default_run(outdir):
    runner = CliRunner()
    outdir = f"{outdir}/out_benchmark"
    result = runner.invoke(benchmark_cli, [
        "--outdir", str(outdir),
        "--num-points", "2",
        "--reps", "1",
        "--min-n", "128",
        "--max-n", "512",
        "--min-knots", "5",
        "--max-knots", "10",
    ])
    assert result.exit_code == 0
    assert "Benchmark complete." in result.output


# def test_bench_plot(outdir):
#     """Run the runtime benchmark with default settings."""
#     outdir = f"{outdir}/out_benchmark"
#     benchmark = RuntimeBenchmark(outdir=outdir)
#     benchmark.plot()

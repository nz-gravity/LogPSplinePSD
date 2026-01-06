import pytest
from click.testing import CliRunner

from log_psplines.benchmark.cli import main as benchmark_cli


@pytest.mark.slow
def test_default_run(outdir):
    runner = CliRunner()
    outdir = f"{outdir}/out_benchmark"
    result = runner.invoke(
        benchmark_cli,
        [
            "--outdir",
            str(outdir),
            "--num-points",
            "1",
            "--reps",
            "1",
            "--min-n",
            "32",
            "--max-n",
            "64",
            "--min-knots",
            "3",
            "--max-knots",
            "4",
            "--verbose",
            "--n-mcmc",
            "5",
        ],
    )

    if result.exit_code != 0 or "Benchmark complete." not in result.output:
        print("CLI Output:\n", result.output)
        print("Exit Code:", result.exit_code)

    assert result.exit_code == 0, "CLI command failed"
    assert "Benchmark complete." in result.output


# def test_bench_plot(outdir):
#     """Run the runtime benchmark with default settings."""
#     outdir = f"{outdir}/out_benchmark"
#     benchmark = RuntimeBenchmark(outdir=outdir)
#     benchmark.plot()

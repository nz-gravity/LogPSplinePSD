import pytest
from click.testing import CliRunner

from log_psplines.benchmark.cli import main as benchmark_cli


def test_default_run(outdir):
    runner = CliRunner()
    outdir = f"{outdir}/out_benchmark"
    result = runner.invoke(benchmark_cli, ["--outdir", str(outdir)])
    assert result.exit_code == 0
    assert "Benchmark complete." in result.output

import click

from .runtime_benchmark import RuntimeBenchmark


@click.command(
    "log_psplines_benchmark", help="Benchmark MCMC runtime performance"
)
@click.option(
    "--analysis",
    type=click.Choice(["data_size", "knots", "samplers", "all"]),
    default="all",
    show_default=True,
    help="Type of analysis to run",
)
@click.option(
    "--outdir",
    default="plots",
    show_default=True,
    help="Output directory",
)
@click.option(
    "--reps",
    type=int,
    default=3,
    show_default=True,
    help="Number of repetitions",
)
@click.option(
    "--plot-only",
    is_flag=True,
    help="Only generate plots",
)
def main(analysis, outdir, reps, plot_only):
    """Benchmark MCMC runtime performance."""
    benchmark = RuntimeBenchmark(outdir)

    if not plot_only:
        if analysis in ["data_size", "all"]:
            benchmark.run_data_size_analysis(reps=reps)

        if analysis in ["knots", "all"]:
            benchmark.run_knots_analysis(reps=reps)

        if analysis in ["samplers", "all"]:
            benchmark.compare_samplers(reps=reps)

    # Generate plots
    benchmark.plot_data_size_results()
    benchmark.plot_knots_results()
    benchmark.plot_sampler_comparison()
    benchmark.generate_summary_report()

    click.echo(f"Benchmark complete. Results saved to {outdir}")


if __name__ == "__main__":
    main()

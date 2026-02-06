import click

from .runtime_benchmark import RuntimeBenchmark


@click.command(
    "log_psplines_benchmark", help="Benchmark MCMC runtime performance"
)
@click.option(
    "--outdir",
    default="plots",
    show_default=True,
    help="Output directory",
)
@click.option(
    "--num-points",
    type=int,
    default=10,
    show_default=True,
    help="Number of points between min and max for analysis",
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
@click.option(
    "--min-n",
    type=int,
    default=128,
    show_default=True,
    help="Minimum data size (in samples) for analysis",
)
@click.option(
    "--max-n",
    type=int,
    default=1024,
    show_default=True,
    help="Maximum data size (in samples) for analysis",
)
@click.option(
    "--min-knots",
    type=int,
    default=5,
    show_default=True,
    help="Minimum number of knots for analysis",
)
@click.option(
    "--max-knots",
    type=int,
    default=30,
    show_default=True,
    help="Maximum number of knots for analysis",
)
@click.option(
    "--verbose",
    is_flag=True,
    default=False,
    help="Enable verbose output",
)
@click.option(
    "--n-mcmc",
    type=int,
    default=2000,
    show_default=True,
    help="Number of MCMC samples to collect (warmup + samples)",
)
@click.option(
    "--sampler",
    type=click.Choice(["nuts", "all"], case_sensitive=False),
    default="nuts",
    show_default=True,
    help="Which samplers to benchmark: 'nuts' or 'all'",
)
def main(
    outdir,
    num_points,
    reps,
    plot_only,
    min_n,
    max_n,
    min_knots,
    max_knots,
    verbose,
    n_mcmc,
    sampler,
):
    """Benchmark MCMC runtime performance."""
    benchmark = RuntimeBenchmark(outdir, verbose=verbose, n_mcmc=n_mcmc)

    if not plot_only:
        benchmark.run_analysis(
            n_points=num_points,
            n_reps=reps,
            min_n=min_n,
            max_n=max_n,
            min_knots=min_knots,
            max_knots=max_knots,
            sampler=sampler,
        )

    benchmark.plot()
    click.echo(f"Benchmark complete. Results saved to {outdir}")


if __name__ == "__main__":
    main()

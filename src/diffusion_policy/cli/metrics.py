"""Metrics aggregation CLI."""

from __future__ import annotations

import click


@click.command(context_settings={"ignore_unknown_options": True, "allow_extra_args": True})
@click.pass_context
def main(ctx: click.Context) -> None:
    """Aggregate multirun metrics."""
    click.echo("Metrics aggregation CLI")
    if ctx.args:
        click.echo("args=" + " ".join(ctx.args))
    click.echo("Implementation migration is pending.")


if __name__ == "__main__":
    main()

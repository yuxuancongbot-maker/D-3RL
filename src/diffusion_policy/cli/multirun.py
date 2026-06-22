"""Multirun CLI."""

from __future__ import annotations

import click


@click.command(context_settings={"ignore_unknown_options": True, "allow_extra_args": True})
@click.pass_context
def main(ctx: click.Context) -> None:
    """Run multi-seed/multi-config experiments."""
    click.echo("Multirun CLI")
    if ctx.args:
        click.echo("args=" + " ".join(ctx.args))
    click.echo("Implementation migration is pending.")


if __name__ == "__main__":
    main()

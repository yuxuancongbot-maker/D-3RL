"""Evaluation CLI for checkpoint evaluation."""

from __future__ import annotations

import click

from diffusion_policy.evaluation.evaluator import run_checkpoint_eval


@click.command(context_settings={"ignore_unknown_options": True, "allow_extra_args": True})
@click.option("--checkpoint", "checkpoint", default=None, help="Checkpoint path to evaluate.")
@click.option("--output-dir", "output_dir", default=None, help="Directory for eval outputs.")
@click.option("--device", default="cuda:0", show_default=True)
@click.option("--steps", default=None, type=int, help="Reserved for future DDIM override support.")
@click.option("--use-ema/--no-use-ema", default=None, help="Override EMA policy selection.")
@click.pass_context
def main(
    ctx: click.Context,
    checkpoint: str | None,
    output_dir: str | None,
    device: str,
    steps: int | None,
    use_ema: bool | None,
) -> None:
    """Evaluate a checkpoint."""
    if steps is not None:
        click.echo("Warning: --steps override is not migrated yet; ignoring for now.")
    if ctx.args:
        click.echo("Warning: extra args are not consumed yet: " + " ".join(ctx.args))
    if checkpoint is None or output_dir is None:
        click.echo("D3RL Diffusion Policy evaluation CLI")
        click.echo("Provide --checkpoint and --output-dir to run evaluation.")
        return
    metrics = run_checkpoint_eval(
        checkpoint=checkpoint,
        output_dir=output_dir,
        device=device,
        use_ema=use_ema,
    )
    click.echo(f"Wrote evaluation with {len(metrics)} metrics to {output_dir}")


if __name__ == "__main__":
    main()

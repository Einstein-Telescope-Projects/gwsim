from __future__ import annotations

import click


@click.command()
@click.option("--config", type=str, help="Configuration file.")
def main(config: str):
    pass

# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Aaron Meza
"""`alpha_assay live-check` diagnostic command.

Evaluates the three live-mode locks (env var, CLI flag, checklist
file) and prints the status to stdout. Exits ``0`` if all three locks
are engaged, ``2`` otherwise. This is a read-only command -- it does
not submit orders, and it does not mutate state.

Intended uses:

- Operator sanity check on the paper-trader host before flipping live.
- CI integration: ``alpha_assay live-check`` in a pre-deploy job to
  confirm the checklist-signed file is staged and the env flag is set
  where expected.
"""

from __future__ import annotations

import click

from alpha_assay.exec.ibkr import check_locks


@click.command("live-check")
@click.option(
    "--live",
    "cli_live",
    is_flag=True,
    default=False,
    help="Engage the CLI-flag lock. Without this, live mode is not possible.",
)
def live_check(cli_live: bool) -> None:
    """Report the status of the three live-mode locks.

    Exits 0 if all three locks are engaged, 2 otherwise. Prints each
    lock's state and, on failure, names every missing lock.
    """
    locks = check_locks(env=None, cli_flag=cli_live, checklist_path=None)

    click.echo(f"env_set        : {'engaged' if locks.env_set else 'MISSING'}")
    click.echo(f"cli_flag       : {'engaged' if locks.cli_flag else 'MISSING'}")
    click.echo(f"checklist_signed: {'engaged' if locks.checklist_signed else 'MISSING'}")

    if locks.all_engaged():
        click.echo("all three locks engaged -> mode=LIVE")
        raise SystemExit(0)

    missing = ", ".join(locks.missing())
    click.echo(f"missing locks: {missing}")
    click.echo("mode=PAPER (live is blocked)")
    raise SystemExit(2)

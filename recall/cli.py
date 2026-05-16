from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

from .session import Session
from .storage import list_sessions, load

app = typer.Typer(help="recall - flight recorder for AI agent sessions")
console = Console()


def _render_session(session: Session, step: bool = False) -> None:
    console.print(
        Panel(
            f"[bold]Session:[/bold] {session.id}\n"
            f"[bold]Provider:[/bold] {session.provider}  "
            f"[bold]Model:[/bold] {session.model or '(unknown)'}\n"
            f"[bold]Started:[/bold] {session.started_at}  "
            f"[bold]Events:[/bold] {len(session.events)}",
            title="[cyan]recall session[/cyan]",
            expand=False,
        )
    )

    for event in session.events:
        if event.type == "request":
            color = "green"
            label = "REQUEST"
        elif event.type == "response":
            color = "blue"
            label = "RESPONSE"
        else:
            color = "red"
            label = "ERROR"

        header = Text()
        header.append(f"[{event.seq}] ", style="dim")
        header.append(label, style=f"bold {color}")
        header.append(f"  {event.timestamp}", style="dim")
        if event.duration_ms is not None:
            header.append(f"  ({event.duration_ms}ms)", style="dim yellow")

        payload_str = json.dumps(event.payload, indent=2)
        syntax = Syntax(payload_str, "json", theme="monokai", word_wrap=True)

        console.print()
        console.print(header)
        console.print(syntax)

        if step and event.seq < len(session.events):
            typer.confirm("  Next event?", default=True)


@app.command()
def play(
    path: Path = typer.Argument(..., help="Path to a session JSON file"),
    step: bool = typer.Option(False, "--step", "-s", help="Step through events one at a time"),
) -> None:
    """Render a recorded session in the terminal."""
    if not path.exists():
        console.print(f"[red]File not found:[/red] {path}")
        raise typer.Exit(1)
    session = load(path)
    _render_session(session, step=step)


@app.command()
def ls(
    dir: Optional[Path] = typer.Option(None, "--dir", "-d", help="Base directory (default: cwd)"),
) -> None:
    """List recorded sessions."""
    paths = list_sessions(dir)
    if not paths:
        console.print("[dim]No sessions recorded yet.[/dim]")
        return

    table = Table(title="Recorded Sessions", show_lines=False)
    table.add_column("File", style="cyan", no_wrap=True)
    table.add_column("Provider")
    table.add_column("Model")
    table.add_column("Started")
    table.add_column("Events", justify="right")

    for p in paths:
        try:
            s = load(p)
            table.add_row(p.name, s.provider, s.model or "-", s.started_at, str(len(s.events)))
        except Exception:
            table.add_row(p.name, "-", "-", "-", "-")

    console.print(table)


@app.command()
def diff(
    session_a: Path = typer.Argument(..., help="First session file"),
    session_b: Path = typer.Argument(..., help="Second session file"),
) -> None:
    """Compare two sessions side by side."""
    for p in (session_a, session_b):
        if not p.exists():
            console.print(f"[red]File not found:[/red] {p}")
            raise typer.Exit(1)

    a = load(session_a)
    b = load(session_b)

    table = Table(title="Session Diff", show_lines=True, expand=True)
    table.add_column(session_a.name, style="green", ratio=1)
    table.add_column(session_b.name, style="blue", ratio=1)

    table.add_row(f"id: {a.id[:8]}", f"id: {b.id[:8]}")
    table.add_row(f"model: {a.model}", f"model: {b.model}")
    table.add_row(f"events: {len(a.events)}", f"events: {len(b.events)}")
    table.add_row(f"started: {a.started_at}", f"started: {b.started_at}")

    console.print(table)

    max_events = max(len(a.events), len(b.events))
    for i in range(max_events):
        ea = a.events[i] if i < len(a.events) else None
        eb = b.events[i] if i < len(b.events) else None

        pa = json.dumps(ea.payload, indent=2) if ea else "(none)"
        pb = json.dumps(eb.payload, indent=2) if eb else "(none)"

        la = f"[{ea.seq}] {ea.type}" if ea else "-"
        lb = f"[{eb.seq}] {eb.type}" if eb else "-"

        row_table = Table.grid(expand=True)
        row_table.add_column(ratio=1)
        row_table.add_column(ratio=1)
        row_table.add_row(
            Panel(Syntax(pa, "json", theme="monokai"), title=f"[green]{la}[/green]"),
            Panel(Syntax(pb, "json", theme="monokai"), title=f"[blue]{lb}[/blue]"),
        )
        console.print(row_table)

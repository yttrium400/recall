from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import typer
from rich.columns import Columns
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

from .providers import anthropic as _anthropic_provider
from .providers import openai as _openai_provider
from .session import Event, Session
from .storage import list_sessions, load


def _estimate_cost(provider: str, model: str, input_tokens: int, output_tokens: int) -> float | None:
    if provider == "openai":
        return _openai_provider.estimate_cost(model, input_tokens, output_tokens)
    return _anthropic_provider.estimate_cost(model, input_tokens, output_tokens)

app = typer.Typer(help="recall - flight recorder for AI agent sessions")
console = Console()


# ---------------------------------------------------------------------------
# play helpers
# ---------------------------------------------------------------------------

def _extract_text(content: list[dict[str, Any]]) -> str:
    parts = []
    for block in content:
        if block.get("type") == "text":
            parts.append(block.get("text", ""))
    return " ".join(parts)


def _render_pretty(session: Session, step: bool) -> None:
    console.print()
    console.print(
        Panel(
            f"[bold cyan]{session.id}[/bold cyan]\n"
            f"[dim]provider:[/dim] {session.provider}  "
            f"[dim]model:[/dim] {session.model or '(unknown)'}  "
            f"[dim]started:[/dim] {session.started_at}",
            title="[bold]recall session[/bold]",
            expand=False,
        )
    )

    requests = [e for e in session.events if e.type == "request"]
    events_by_seq = {e.seq: e for e in session.events}
    pairs: list[tuple[Event, Event | None]] = []
    for req in requests:
        following = events_by_seq.get(req.seq + 1)
        pairs.append((req, following if following and following.type != "request" else None))

    displayed_count = 0  # number of messages already shown to the user

    for call_idx, (req, resp) in enumerate(pairs):
        console.print()
        console.print(Rule(f"[bold]Call #{call_idx + 1}[/bold]  [dim]{req.timestamp}[/dim]", style="dim"))

        payload = req.payload
        all_messages: list[dict[str, Any]] = payload.get("messages", [])
        system = payload.get("system", "")

        if call_idx == 0 and system:
            system_text = system if isinstance(system, str) else json.dumps(system)
            console.print(Panel(system_text, title="[dim]system[/dim]", border_style="dim", expand=False))

        # Show only messages added since the last call (skip already-displayed history)
        new_messages = all_messages[displayed_count:]
        displayed_count = len(all_messages) + 1  # +1 for the assistant turn we're about to show

        for msg in new_messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            color = "green" if role == "user" else "blue"
            label = f"[bold {color}]{role}[/bold {color}]"
            _render_message_content(label, content)

        if resp:
            if resp.type == "error":
                console.print(
                    Panel(
                        f"[red]{resp.payload.get('type', 'Error')}[/red]: {resp.payload.get('error', '')}",
                        title="[red bold]error[/red bold]",
                        border_style="red",
                    )
                )
            else:
                resp_content = _extract_resp_content(resp.payload)
                _render_message_content("[bold blue]assistant[/bold blue]", resp_content)

                usage = resp.payload.get("usage", {})
                in_tok = usage.get("input_tokens", 0)
                out_tok = usage.get("output_tokens", 0)
                cost = _estimate_cost(session.provider, session.model, in_tok, out_tok)
                cost_str = f"  ~${cost:.6f}" if cost is not None else ""
                dur_str = f"  {resp.duration_ms}ms" if resp.duration_ms is not None else ""
                console.print(
                    f"  [dim]tokens: {in_tok} in / {out_tok} out{cost_str}{dur_str}[/dim]"
                )

        if step and call_idx < len(pairs) - 1:
            typer.confirm("  Next call?", default=True)


def _format_content_blocks(blocks: list[dict[str, Any]]) -> str:
    """Return plain-text representation of content blocks (used for markdown export)."""
    parts = []
    for block in blocks:
        btype = block.get("type", "")
        if btype == "text":
            parts.append(block.get("text", ""))
        elif btype == "tool_use":
            name = block.get("name", "")
            inp = json.dumps(block.get("input", {}))
            parts.append(f"[tool_use: {name}] {inp}")
        elif btype == "tool_result":
            content = block.get("content", "")
            if isinstance(content, list):
                content = _extract_text(content)
            parts.append(f"[tool_result] {content}")
        else:
            parts.append(json.dumps(block))
    return "\n    ".join(parts)


def _render_message_content(label: str, content: Any) -> None:
    """Render a message with Rich panel visualization for tool calls."""
    if isinstance(content, str):
        console.print(f"  {label}: {content}")
        return
    if not isinstance(content, list):
        console.print(f"  {label}: {content}")
        return

    text_parts = [b.get("text", "") for b in content if b.get("type") == "text"]
    has_tools = any(b.get("type") in ("tool_use", "tool_result") for b in content)

    if text_parts:
        console.print(f"  {label}: {' '.join(text_parts)}")
    elif has_tools:
        console.print(f"  {label}:")
    else:
        console.print(f"  {label}:")

    for block in content:
        btype = block.get("type", "")
        if btype == "tool_use":
            name = block.get("name", "")
            tool_id = block.get("id", "")
            inp = json.dumps(block.get("input", {}), indent=2)
            id_suffix = f"  [dim]{tool_id}[/dim]" if tool_id else ""
            console.print(Panel(
                Syntax(inp, "json", theme="monokai", word_wrap=True),
                title=f"[bold yellow]tool_use[/bold yellow]: [cyan]{name}[/cyan]{id_suffix}",
                border_style="yellow",
                padding=(0, 1),
                expand=False,
            ))
        elif btype == "tool_result":
            tool_content = block.get("content", "")
            if isinstance(tool_content, list):
                tool_content = _extract_text(tool_content)
            tool_id = block.get("tool_use_id", "")
            id_suffix = f"  [dim]{tool_id}[/dim]" if tool_id else ""
            console.print(Panel(
                str(tool_content),
                title=f"[bold green]tool_result[/bold green]{id_suffix}",
                border_style="green",
                padding=(0, 1),
                expand=False,
            ))


def _render_raw(session: Session, step: bool) -> None:
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
            color, label = "green", "REQUEST"
        elif event.type == "response":
            color, label = "blue", "RESPONSE"
        else:
            color, label = "red", "ERROR"

        header = Text()
        header.append(f"[{event.seq}] ", style="dim")
        header.append(label, style=f"bold {color}")
        header.append(f"  {event.timestamp}", style="dim")
        if event.duration_ms is not None:
            header.append(f"  ({event.duration_ms}ms)", style="dim yellow")

        console.print()
        console.print(header)
        console.print(Syntax(json.dumps(event.payload, indent=2), "json", theme="monokai", word_wrap=True))

        if step and event.seq < len(session.events):
            typer.confirm("  Next event?", default=True)


# ---------------------------------------------------------------------------
# CLI commands
# ---------------------------------------------------------------------------

@app.command()
def play(
    path: Path = typer.Argument(..., help="Path to a session JSON file"),
    step: bool = typer.Option(False, "--step", "-s", help="Step through calls one at a time"),
    raw: bool = typer.Option(False, "--raw", "-r", help="Show raw JSON events instead of conversation view"),
) -> None:
    """Render a recorded session in the terminal."""
    if not path.exists():
        console.print(f"[red]File not found:[/red] {path}")
        raise typer.Exit(1)
    session = load(path)
    if raw:
        _render_raw(session, step=step)
    else:
        _render_pretty(session, step=step)


@app.command()
def stats(
    path: Path = typer.Argument(..., help="Path to a session JSON file"),
) -> None:
    """Show token usage, cost estimate, and latency for a session."""
    if not path.exists():
        console.print(f"[red]File not found:[/red] {path}")
        raise typer.Exit(1)

    session = load(path)
    responses = [e for e in session.events if e.type == "response"]
    errors = [e for e in session.events if e.type == "error"]

    total_in = total_out = total_cache_read = total_cache_write = 0
    total_ms = 0
    latencies: list[int] = []

    for r in responses:
        usage = r.payload.get("usage", {})
        # Anthropic: input_tokens/output_tokens; OpenAI: prompt_tokens/completion_tokens
        total_in += usage.get("input_tokens", 0) or usage.get("prompt_tokens", 0)
        total_out += usage.get("output_tokens", 0) or usage.get("completion_tokens", 0)
        total_cache_read += usage.get("cache_read_input_tokens", 0)
        total_cache_write += usage.get("cache_creation_input_tokens", 0)
        if r.duration_ms is not None:
            total_ms += r.duration_ms
            latencies.append(r.duration_ms)

    cost = _estimate_cost(session.provider, session.model, total_in, total_out)

    table = Table(title=f"Stats: {path.name}", show_header=False, box=None, padding=(0, 2))
    table.add_column(style="dim", no_wrap=True)
    table.add_column()

    table.add_row("Session ID", session.id)
    table.add_row("Model", session.model or "(unknown)")
    table.add_row("Provider", session.provider)
    table.add_row("Started", session.started_at)
    table.add_row("API calls", str(len(responses)))
    table.add_row("Errors", str(len(errors)))
    table.add_row("Input tokens", f"{total_in:,}")
    table.add_row("Output tokens", f"{total_out:,}")
    if total_cache_read:
        table.add_row("Cache read tokens", f"{total_cache_read:,}")
    if total_cache_write:
        table.add_row("Cache write tokens", f"{total_cache_write:,}")
    if cost is not None:
        table.add_row("Estimated cost", f"${cost:.6f}")
    if latencies:
        avg_ms = total_ms // len(latencies)
        table.add_row("Total latency", f"{total_ms:,}ms")
        table.add_row("Avg latency / call", f"{avg_ms}ms")
        table.add_row("Slowest call", f"{max(latencies)}ms")

    console.print()
    console.print(table)
    console.print()


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
    table.add_column("Model")
    table.add_column("Calls", justify="right")
    table.add_column("Tokens", justify="right")
    table.add_column("Cost", justify="right")
    table.add_column("Started")

    for p in paths:
        try:
            s = load(p)
            responses = [e for e in s.events if e.type == "response"]
            calls = len(responses)
            in_tok = sum(
                (e.payload.get("usage", {}).get("input_tokens", 0) or e.payload.get("usage", {}).get("prompt_tokens", 0))
                for e in responses
            )
            out_tok = sum(
                (e.payload.get("usage", {}).get("output_tokens", 0) or e.payload.get("usage", {}).get("completion_tokens", 0))
                for e in responses
            )
            cost = _estimate_cost(s.provider, s.model, in_tok, out_tok)
            cost_str = f"${cost:.5f}" if cost is not None else "-"
            tok_str = f"{in_tok + out_tok:,}" if (in_tok or out_tok) else "-"
            table.add_row(p.name, s.model or "-", str(calls), tok_str, cost_str, s.started_at)
        except Exception:
            table.add_row(p.name, "-", "-", "-", "-", "-")

    console.print(table)


@app.command()
def export(
    path: Path = typer.Argument(..., help="Path to a session JSON file"),
    format: str = typer.Option("markdown", "--format", "-f", help="Output format (markdown)"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Write to file instead of stdout"),
) -> None:
    """Export a session to a shareable format (default: markdown)."""
    if not path.exists():
        console.print(f"[red]File not found:[/red] {path}")
        raise typer.Exit(1)

    if format != "markdown":
        console.print(f"[red]Unsupported format:[/red] {format}. Only 'markdown' is supported.")
        raise typer.Exit(1)

    session = load(path)
    md = _render_markdown(session)

    if output:
        output.write_text(md, encoding="utf-8")
        console.print(f"[green]Exported to[/green] {output}")
    else:
        print(md)


def _extract_resp_content(payload: dict[str, Any]) -> Any:
    """Return the assistant content from a response payload (handles Anthropic and OpenAI)."""
    content = payload.get("content")
    if content is not None:
        return content
    choices = payload.get("choices", [])
    if choices:
        msg_content = choices[0].get("message", {}).get("content", "")
        return msg_content if msg_content else []
    return []


def _render_markdown(session: Session) -> str:
    lines: list[str] = []

    lines.append("# Recall Session Report\n")
    lines.append(f"**Session ID:** {session.id}  ")
    lines.append(f"**Provider:** {session.provider}  ")
    lines.append(f"**Model:** {session.model or '(unknown)'}  ")
    lines.append(f"**Started:** {session.started_at}  ")
    lines.append("")

    requests = [e for e in session.events if e.type == "request"]
    events_by_seq = {e.seq: e for e in session.events}
    pairs: list[tuple[Any, Any]] = []
    for req in requests:
        following = events_by_seq.get(req.seq + 1)
        pairs.append((req, following if following and following.type != "request" else None))

    displayed_count = 0
    total_in = total_out = total_ms = 0
    costs: list[float] = []

    for call_idx, (req, resp) in enumerate(pairs):
        lines.append("---\n")
        ts = req.timestamp.split("T")[-1].rstrip("Z")
        lines.append(f"## Call #{call_idx + 1} — {ts}\n")

        payload = req.payload
        all_messages: list[dict[str, Any]] = payload.get("messages", [])
        system = payload.get("system", "")

        if call_idx == 0 and system:
            system_text = system if isinstance(system, str) else json.dumps(system)
            lines.append("**System:**\n")
            for sline in system_text.splitlines():
                lines.append(f"> {sline}")
            lines.append("")

        new_messages = all_messages[displayed_count:]
        displayed_count = len(all_messages) + 1

        for msg in new_messages:
            role = msg.get("role", "").capitalize()
            content = msg.get("content", "")
            lines.extend(_md_message(role, content))

        if resp:
            if resp.type == "error":
                lines.append(f"**Error:** `{resp.payload.get('type', 'Error')}`: {resp.payload.get('error', '')}\n")
            else:
                resp_content = _extract_resp_content(resp.payload)
                lines.extend(_md_message("Assistant", resp_content))

                usage = resp.payload.get("usage", {})
                in_tok = usage.get("input_tokens", 0) or usage.get("prompt_tokens", 0)
                out_tok = usage.get("output_tokens", 0) or usage.get("completion_tokens", 0)
                total_in += in_tok
                total_out += out_tok
                if resp.duration_ms:
                    total_ms += resp.duration_ms
                cost = _estimate_cost(session.provider, session.model, in_tok, out_tok)
                cost_str = f" | ~${cost:.6f}" if cost is not None else ""
                if cost is not None:
                    costs.append(cost)
                dur_str = f" | {resp.duration_ms}ms" if resp.duration_ms else ""
                lines.append(f"> *{in_tok} in / {out_tok} out{cost_str}{dur_str}*\n")

    lines.append("---\n")
    lines.append("## Summary\n")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    lines.append(f"| Total calls | {len(pairs)} |")
    lines.append(f"| Input tokens | {total_in:,} |")
    lines.append(f"| Output tokens | {total_out:,} |")
    if costs:
        lines.append(f"| Estimated cost | ${sum(costs):.6f} |")
    if total_ms:
        lines.append(f"| Total latency | {total_ms:,}ms |")
    lines.append("")

    return "\n".join(lines)


def _md_message(role: str, content: Any) -> list[str]:
    lines: list[str] = []
    if isinstance(content, str):
        lines.append(f"**{role}:** {content}\n")
        return lines

    if not isinstance(content, list):
        lines.append(f"**{role}:** {content}\n")
        return lines

    text_parts = [b.get("text", "") for b in content if b.get("type") == "text"]
    has_tools = any(b.get("type") in ("tool_use", "tool_result") for b in content)

    if text_parts:
        lines.append(f"**{role}:** {' '.join(text_parts)}\n")
    elif has_tools:
        lines.append(f"**{role}:**\n")

    for block in content:
        btype = block.get("type", "")
        if btype == "tool_use":
            name = block.get("name", "")
            tool_id = block.get("id", "")
            inp = json.dumps(block.get("input", {}), indent=2)
            id_note = f" `{tool_id}`" if tool_id else ""
            lines.append(f"**Tool call:** `{name}`{id_note}\n")
            lines.append("```json")
            lines.append(inp)
            lines.append("```\n")
        elif btype == "tool_result":
            tool_content = block.get("content", "")
            if isinstance(tool_content, list):
                tool_content = _extract_text(tool_content)
            tool_id = block.get("tool_use_id", "")
            id_note = f" `{tool_id}`" if tool_id else ""
            lines.append(f"**Tool result:**{id_note}\n")
            lines.append("```")
            lines.append(str(tool_content))
            lines.append("```\n")

    return lines


@app.command()
def diff(
    session_a: Path = typer.Argument(..., help="First session file"),
    session_b: Path = typer.Argument(..., help="Second session file"),
) -> None:
    """Compare two sessions and show where they diverge."""
    for p in (session_a, session_b):
        if not p.exists():
            console.print(f"[red]File not found:[/red] {p}")
            raise typer.Exit(1)

    a = load(session_a)
    b = load(session_b)

    # Header summary
    header = Table(show_header=True, box=None, padding=(0, 3))
    header.add_column("", style="dim")
    header.add_column(f"[green]{session_a.name}[/green]")
    header.add_column(f"[blue]{session_b.name}[/blue]")
    header.add_row("model", a.model or "-", b.model or "-")
    header.add_row("calls", str(sum(1 for e in a.events if e.type == "request")),
                            str(sum(1 for e in b.events if e.type == "request")))
    header.add_row("started", a.started_at, b.started_at)
    console.print()
    console.print(header)

    # Find divergence point
    a_reqs = [e for e in a.events if e.type == "request"]
    b_reqs = [e for e in b.events if e.type == "request"]
    a_resps = [e for e in a.events if e.type == "response"]
    b_resps = [e for e in b.events if e.type == "response"]

    diverged_at: int | None = None
    for i, (ra, rb) in enumerate(zip(a_resps, b_resps)):
        if ra.payload != rb.payload:
            diverged_at = i
            break

    if diverged_at is None and len(a_resps) != len(b_resps):
        diverged_at = min(len(a_resps), len(b_resps))

    if diverged_at is not None:
        console.print(f"\n[yellow]Sessions diverge at call #{diverged_at + 1}[/yellow]\n")
    else:
        console.print("\n[green]Sessions are identical in all responses[/green]\n")

    max_calls = max(len(a_resps), len(b_resps))
    for i in range(max_calls):
        ra = a_resps[i] if i < len(a_resps) else None
        rb = b_resps[i] if i < len(b_resps) else None

        pa = json.dumps(ra.payload, indent=2) if ra else "(none)"
        pb = json.dumps(rb.payload, indent=2) if rb else "(none)"

        marker = " [yellow]*[/yellow]" if i == diverged_at else ""
        title_a = f"[green]call #{i + 1}[/green]{marker}"
        title_b = f"[blue]call #{i + 1}[/blue]{marker}"

        left = Panel(Syntax(pa, "json", theme="monokai", word_wrap=True), title=title_a)
        right = Panel(Syntax(pb, "json", theme="monokai", word_wrap=True), title=title_b)
        console.print(Columns([left, right]))

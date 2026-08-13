#!/usr/bin/env python3
"""
show_progress.py
================
Clean live progress monitor for Tau2-Bench simulation runs.

Provides smooth, in-place live terminal updates (using Rich Live) during live watch,
and 100% full scrollable terminal printing when run with --no-watch.

Usage
-----
  python show_progress.py                          # live watch mode (auto-discovers latest run)
  python show_progress.py --no-watch --verbose     # full 100% scrollable transcript
  python show_progress.py --watch --verbose        # live step stream (last 8 steps in live view)
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

import typer
from rich import box
from rich.console import Console
from rich.console import Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table

# Import Tau2 simulation data model
try:
    from tau2.data_model.simulation import Results
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
    from tau2.data_model.simulation import Results

app = typer.Typer(
    help="Clean live progress monitor for Tau2-Bench simulation runs.",
    add_completion=False,
)
console = Console()


def interactive_select_results_file(start_dir: Path) -> Path:
    """Interactively navigate directories and select a simulation run results.json."""
    current_dir = start_dir

    while True:
        has_results = (current_dir / "results.json").exists()
        subdirs = [p for p in current_dir.iterdir() if p.is_dir() and not p.name.startswith(".")]
        subdirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)

        if not subdirs:
            if has_results:
                return current_dir / "results.json"
            if (current_dir / "simulations").exists() or list(current_dir.glob("*.json")):
                return current_dir / "results.json"
            console.print(f"[bold red]No simulation results found in:[/] {current_dir}")
            raise typer.Exit(1)

        console.print(f"\n[bold cyan]📁 Select a simulation folder from:[/] [yellow]{current_dir}[/yellow]\n")

        options: list[tuple[str, str, Optional[Path]]] = []
        if has_results:
            options.append(("0", f"View results in current folder ({current_dir.name})", None))

        for idx, sdir in enumerate(subdirs, 1):
            options.append((str(idx), sdir.name, sdir))

        for opt_key, label, path_obj in options:
            if opt_key == "0":
                console.print(f"  [[bold green]0[/bold green]] [bold green]{label}[/bold green]")
            else:
                mtime_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(path_obj.stat().st_mtime))
                badge = "[dim cyan](run folder)[/dim cyan]" if (path_obj / "results.json").exists() else "[dim yellow](group folder)[/dim yellow]"
                console.print(f"  [[bold yellow]{opt_key:>{len(str(len(options)))}}[/bold yellow]] [bold]{label}[/bold] {badge} [dim]— {mtime_str}[/dim]")

        console.print()
        prompt_parts = []
        if current_dir != start_dir:
            prompt_parts.append("'b' for back")
        prompt_parts.append("'q' to quit")
        prompt_str = f"Enter number ({', '.join(prompt_parts)}): "

        try:
            choice = input(prompt_str).strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[yellow]Selection cancelled.[/yellow]")
            raise typer.Exit(0)

        if choice.lower() == "q":
            raise typer.Exit(0)
        if choice.lower() == "b" and current_dir != start_dir:
            current_dir = current_dir.parent
            continue

        selected = next((opt for opt in options if opt[0] == choice), None)
        if selected:
            _, _, selected_path = selected
            if selected_path is None:
                return current_dir / "results.json"

            child_subdirs = [p for p in selected_path.iterdir() if p.is_dir() and not p.name.startswith(".")]
            if not child_subdirs and (selected_path / "results.json").exists():
                return selected_path / "results.json"

            current_dir = selected_path
        else:
            console.print("[bold red]Invalid choice. Please select a valid number from the list.[/bold red]")


def resolve_results_file(path_arg: Optional[Path]) -> Path:
    """Resolve argument or interactively select results.json in data/simulations/."""
    if path_arg:
        if path_arg.is_file() and path_arg.name.endswith(".json"):
            return path_arg
        if path_arg.is_dir():
            if sys.stdin.isatty():
                return interactive_select_results_file(path_arg)
            candidate = path_arg / "results.json"
            if candidate.exists():
                return candidate

    sim_base = Path(__file__).resolve().parent / "data" / "simulations"
    if not sim_base.exists():
        sim_base = Path("data/simulations")

    if not sim_base.exists():
        console.print("[bold red]Error:[/] data/simulations/ directory does not exist.")
        raise typer.Exit(1)

    if sys.stdin.isatty():
        return interactive_select_results_file(sim_base)

    candidates = sorted(
        sim_base.rglob("results.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if candidates:
        return candidates[0]

    console.print(
        "[bold red]Error:[/] No results.json found in data/simulations/. "
        "Pass a directory or file path explicitly."
    )
    raise typer.Exit(1)


def get_file_mtime_str(path: Path) -> str:
    if not path.exists():
        return "N/A"
    return time.strftime("%H:%M:%S", time.localtime(path.stat().st_mtime))


def format_json_snippet(data: str | dict, max_len: int = 200) -> str:
    """Format json text nicely with truncation."""
    if isinstance(data, dict):
        text_str = json.dumps(data)
    else:
        text_str = str(data)
    if len(text_str) > max_len:
        return text_str[:max_len] + "..."
    return text_str


def build_dashboard_group(
    results_path: Path, verbose: bool = False, last_only: bool = False, tail_count: int = 8, watch_mode: bool = True
) -> Group:
    """Build Rich renderable Group for clean display."""
    renderables = []

    if not results_path.exists():
        renderables.append(
            Panel(
                f"Waiting for benchmark run to initialize...\nPath: {results_path}",
                title="Tau2 Benchmark Live Monitor",
                border_style="yellow",
                box=box.ROUNDED,
            )
        )
        return Group(*renderables)

    sim_dir = results_path.parent / "simulations"

    try:
        results = Results.load(results_path)
    except Exception as e:
        renderables.append(
            Panel(
                f"Reading results file... ({e})",
                title="Tau2 Benchmark Live Monitor",
                border_style="yellow",
                box=box.ROUNDED,
            )
        )
        return Group(*renderables)

    info = results.info or {}
    simulations = list(results.simulations) if results.simulations else []

    if not simulations and sim_dir.exists():
        for sim_file in sorted(sim_dir.glob("*.json")):
            try:
                with open(sim_file) as f:
                    sim_data = json.load(f)
                    if isinstance(sim_data, dict) and "task_id" in sim_data:
                        simulations.append(sim_data)
            except Exception:
                pass

    domain = "MOCK"
    if hasattr(info, "environment_info") and info.environment_info and getattr(info.environment_info, "domain_name", None):
        domain = info.environment_info.domain_name
    elif hasattr(info, "domain") and info.domain:
        domain = info.domain
    elif hasattr(results, "domain") and results.domain:
        domain = results.domain
    else:
        parts = results_path.parent.name.split("_")
        if len(parts) >= 3:
            domain = parts[2]

    domain = str(domain).upper()
    agent_name = str(getattr(info, "agent", "rac_planner"))
    agent_llm = str(getattr(info, "llm_agent", "openai/gpt-5.6-luna"))
    user_llm = str(getattr(info, "llm_user", "openai/gpt-5.6-luna"))
    mtime = get_file_mtime_str(results_path)

    completed_sims = len(simulations)
    successful_sims = 0
    for s in simulations:
        reward = None
        if hasattr(s, "reward_info") and s.reward_info:
            reward = getattr(s.reward_info, "reward", None)
        elif isinstance(s, dict) and "reward_info" in s and s["reward_info"]:
            reward = s["reward_info"].get("reward")
        if reward == 1.0:
            successful_sims += 1

    pass_rate = (successful_sims / completed_sims * 100) if completed_sims > 0 else 0.0

    header_content = (
        f"Domain: {domain}  |  Agent: {agent_name} ({agent_llm})  |  User: {user_llm}\n"
        f"Completed Tasks: {completed_sims}  |  Pass Rate: [{ 'green' if pass_rate >= 80 else 'yellow' if pass_rate >= 50 else 'red' }]{pass_rate:.1f}% ({successful_sims}/{completed_sims})[/]  |  Updated: {mtime} (Live: {time.strftime('%H:%M:%S')})"
    )
    renderables.append(
        Panel(
            header_content,
            title="Tau2 Benchmark Progress Monitor",
            border_style="cyan",
            box=box.ROUNDED,
        )
    )

    table = Table(
        show_header=True,
        header_style="bold cyan",
        box=box.SIMPLE,
        expand=True,
    )
    table.add_column("Task ID", width=24)
    table.add_column("Status", justify="center", width=14)
    table.add_column("Reward", justify="right", width=10)
    table.add_column("Steps", justify="right", width=8)
    table.add_column("Tool Calls", justify="right", width=12)

    live_trace = None
    possible_trace_paths = [
        results_path.parent / "live_trace.json",
        results_path.parent.parent / "live_trace.json",
        Path("data/simulations/live_trace.json"),
    ]
    trace_path = None
    for p in possible_trace_paths:
        if p.exists():
            trace_path = p
            break

    if trace_path and trace_path.exists():
        try:
            if time.time() - trace_path.stat().st_mtime < 180:
                with open(trace_path) as f:
                    live_trace = json.load(f)
        except Exception:
            pass

    for sim in simulations:
        task_id = sim.task_id if hasattr(sim, "task_id") else sim.get("task_id", "N/A")
        reward_val = 0.0
        if hasattr(sim, "reward_info") and sim.reward_info:
            reward_val = getattr(sim.reward_info, "reward", 0.0)
        elif isinstance(sim, dict) and "reward_info" in sim and sim["reward_info"]:
            reward_val = sim["reward_info"].get("reward", 0.0)

        status_str = "[green]PASSED[/]" if reward_val == 1.0 else "[red]FAILED[/]"

        msgs = getattr(sim, "messages", None) or (sim.get("messages") if isinstance(sim, dict) else None) or []
        msg_count = len(msgs)
        tool_count = 0
        for m in msgs:
            t_calls = getattr(m, "tool_calls", None) or (m.get("tool_calls") if isinstance(m, dict) else None)
            if t_calls:
                tool_count += len(t_calls)

        table.add_row(
            str(task_id),
            status_str,
            f"{reward_val:.2f}",
            str(msg_count),
            str(tool_count),
        )

    is_active_running = False
    if live_trace and trace_path.exists() and results_path.exists():
        # Active only if live_trace was modified AFTER results.json was last updated and within last 60 seconds
        if (trace_path.stat().st_mtime > results_path.stat().st_mtime) and (time.time() - trace_path.stat().st_mtime < 60):
            is_active_running = True

    if is_active_running and live_trace and live_trace.get("messages"):
        active_msgs = live_trace.get("messages", [])
        active_tools = sum(1 for m in active_msgs if m.get("tool_calls"))
        plan_label = live_trace.get("current_plan_name") or "Direct ReAct Execution"
        table.add_row(
            f"Active Task ({plan_label})",
            "[yellow]RUNNING...[/]",
            "0.00",
            str(len(active_msgs)),
            str(active_tools),
        )
    elif not simulations:
        table.add_row("Task in progress...", "[yellow]RUNNING...[/]", "0.00", "0", "0")

    renderables.append(table)

    if verbose:
        if is_active_running and live_trace and live_trace.get("messages"):
            active_msgs = live_trace.get("messages", [])
            display_msgs = active_msgs[-tail_count:] if (watch_mode and len(active_msgs) > tail_count) else active_msgs
            start_offset = len(active_msgs) - len(display_msgs)

            t_panels = []
            c_info = live_trace.get("candidates_info", [])
            if c_info:
                p_table = Table(title="RAC Pareto Plan Candidates", box=box.SIMPLE, show_header=True, header_style="bold green")
                p_table.add_column("Candidate Plan", width=22)
                p_table.add_column("Cost", justify="right", width=8)
                p_table.add_column("Risk", justify="right", width=8)
                p_table.add_column("Status", justify="center", width=12)
                p_table.add_column("Strategy Summary", width=45)
                for c in c_info:
                    sel_str = "[bold green]SELECTED[/]" if c.get("selected") else "[dim]Option[/]"
                    p_table.add_row(
                        str(c.get("name", "")),
                        f"{c.get('cost', 0.0):.2f}",
                        f"{c.get('risk', 0.0):.2f}",
                        sel_str,
                        format_json_snippet(c.get("strategy_summary", ""), max_len=120),
                    )
                t_panels.append(p_table)

            t_panels.append(f"[yellow]Active Task Step Stream ({live_trace.get('current_plan_name', 'RAC Plan')}) - Updated: {live_trace.get('updated_at', '')}[/yellow]")
            if start_offset > 0:
                t_panels.append(f"[dim]... ({start_offset} earlier steps hidden in live view. Run with --no-watch to view full transcript) ...[/dim]")

            for rel_idx, msg in enumerate(display_msgs):
                idx = start_offset + rel_idx
                role_str = str(msg.get("role", ""))
                content = str(msg.get("content", ""))
                t_calls = msg.get("tool_calls")

                if role_str == "user":
                    t_panels.append(
                        Panel(
                            content,
                            title=f"User (Step {idx})",
                            border_style="blue",
                            box=box.ROUNDED,
                        )
                    )
                elif role_str == "assistant":
                    if content:
                        t_panels.append(
                            Panel(
                                content,
                                title=f"RAC Agent (Step {idx})",
                                border_style="green",
                                box=box.ROUNDED,
                            )
                        )
                    if t_calls:
                        calls_text = []
                        for tc in t_calls:
                            calls_text.append(f"Tool: {tc.get('name')}\nArgs: {format_json_snippet(tc.get('arguments', {}))}")
                        t_panels.append(
                            Panel(
                                "\n---\n".join(calls_text),
                                title=f"Tool Call (Step {idx})",
                                border_style="yellow",
                                box=box.ROUNDED,
                            )
                        )
                elif role_str in ("tool", "environment"):
                    snippet = format_json_snippet(content, max_len=250)
                    t_panels.append(
                        Panel(
                            snippet,
                            title=f"Tool Result (Step {idx})",
                            border_style="dim",
                            box=box.ROUNDED,
                        )
                    )
            renderables.append(Group(*t_panels))

        if simulations:
            target_sims = [simulations[-1]] if last_only else simulations
            for sim in target_sims:
                task_id = sim.task_id if hasattr(sim, "task_id") else sim.get("task_id", "N/A")
                reward_val = 0.0
                if hasattr(sim, "reward_info") and sim.reward_info:
                    reward_val = getattr(sim.reward_info, "reward", 0.0)
                elif isinstance(sim, dict) and "reward_info" in sim and sim["reward_info"]:
                    reward_val = sim["reward_info"].get("reward", 0.0)

                r_color = "green" if reward_val == 1.0 else "red"
                t_panels = []
                t_panels.append(f"[{r_color}]Transcript for Task: {task_id} (Reward: {reward_val})[/{r_color}]")

                msgs = getattr(sim, "messages", None) or (sim.get("messages") if isinstance(sim, dict) else None) or []
                display_sim_msgs = msgs[-tail_count:] if (watch_mode and len(msgs) > tail_count) else msgs
                sim_start_offset = len(msgs) - len(display_sim_msgs)

                if sim_start_offset > 0:
                    t_panels.append(f"[dim]... ({sim_start_offset} earlier steps hidden in live view. Run with --no-watch to view full transcript) ...[/dim]")

                for rel_idx, msg in enumerate(display_sim_msgs):
                    idx = sim_start_offset + rel_idx
                    role = getattr(msg, "role", None) or (msg.get("role") if isinstance(msg, dict) else "")
                    role_str = role.value if hasattr(role, "value") else str(role)
                    content = getattr(msg, "content", None) or (msg.get("content") if isinstance(msg, dict) else "")

                    if role_str == "user":
                        t_panels.append(
                            Panel(
                                str(content),
                                title=f"User (Step {idx})",
                                border_style="blue",
                                box=box.ROUNDED,
                            )
                        )
                    elif role_str == "assistant":
                        if content:
                            t_panels.append(
                                Panel(
                                    str(content),
                                    title=f"RAC Agent (Step {idx})",
                                    border_style="green",
                                    box=box.ROUNDED,
                                )
                            )
                        t_calls = getattr(msg, "tool_calls", None) or (msg.get("tool_calls") if isinstance(msg, dict) else None)
                        if t_calls:
                            calls_text = []
                            for tc in t_calls:
                                tc_name = getattr(tc, "name", None) or (tc.get("name") if isinstance(tc, dict) else "tool")
                                tc_args = getattr(tc, "arguments", None) or (tc.get("arguments") if isinstance(tc, dict) else {})
                                calls_text.append(f"Tool: {tc_name}\nArgs: {format_json_snippet(tc_args)}")
                            t_panels.append(
                                Panel(
                                    "\n---\n".join(calls_text),
                                    title=f"Tool Call (Step {idx})",
                                    border_style="yellow",
                                    box=box.ROUNDED,
                                )
                            )
                    elif role_str in ("tool", "environment"):
                        snippet = format_json_snippet(content, max_len=250)
                        t_panels.append(
                            Panel(
                                snippet,
                                title=f"Tool Result (Step {idx})",
                                border_style="dim",
                                box=box.ROUNDED,
                            )
                        )
                renderables.append(Group(*t_panels))

    return Group(*renderables)


@app.command()
def main(
    path: Optional[Path] = typer.Argument(
        None,
        help="Path to run directory or results.json file. If omitted, auto-discovers latest run in data/simulations/.",
    ),
    watch: bool = typer.Option(
        True,
        "--watch/--no-watch",
        "-w/-nw",
        help="Enable live in-place updating (default: True).",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Display detailed turn-by-turn conversation transcripts.",
    ),
    last_only: bool = typer.Option(
        False,
        "--last-only",
        "-l",
        help="Only display detailed transcript for the most recent task.",
    ),
    tail: int = typer.Option(
        3,
        "--tail",
        "-t",
        help="Number of latest steps to show during live watch mode (default: 3).",
    ),
    refresh_rate: float = typer.Option(
        0.5,
        "--refresh-rate",
        "-r",
        help="Refresh interval in seconds for live updating (default: 0.5s).",
    ),
) -> None:
    """Clean live progress monitor with in-place updates for Tau2 Benchmark runs."""
    results_path = resolve_results_file(path)

    if watch:
        console.print(f"[dim]Live watching results at: {results_path} (Ctrl+C to stop, run with --no-watch for 100% full scrollable view)[/]\n")
        try:
            with Live(
                build_dashboard_group(results_path, verbose=verbose, last_only=last_only, tail_count=tail, watch_mode=True),
                console=console,
                refresh_per_second=int(1.0 / refresh_rate),
                auto_refresh=True,
            ) as live:
                while True:
                    live.update(build_dashboard_group(results_path, verbose=verbose, last_only=last_only, tail_count=tail, watch_mode=True))
                    time.sleep(refresh_rate)
        except KeyboardInterrupt:
            console.print("\n[yellow]Live watch stopped.[/]")
    else:
        # Static print mode: prints full un-truncated scrollable text directly to terminal buffer
        group_obj = build_dashboard_group(results_path, verbose=verbose, last_only=last_only, tail_count=9999, watch_mode=False)
        console.print(group_obj)


if __name__ == "__main__":
    app()

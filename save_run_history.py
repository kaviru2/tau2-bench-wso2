"""CSV & Structured Log History Exporter for Tau2 Benchmark Runs.

Appends all completed run stats, rewards, database checks, communication checks,
and log file paths to `data/simulations/run_history.csv` for easy analysis.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from tau2.data_model.simulation import Results


def update_run_history_csv(
    sim_base_dir: str | Path = "data/simulations",
    csv_out_path: str | Path = "data/simulations/run_history.csv",
) -> Path:
    """Scan all simulation run results in sim_base_dir and export/append them to CSV."""
    base = Path(sim_base_dir)
    csv_file = Path(csv_out_path)
    csv_file.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "Timestamp",
        "Run_Folder",
        "Domain",
        "Task_ID",
        "Trial",
        "Agent",
        "Reward",
        "DB_Match",
        "DB_Reward",
        "Action_Reward",
        "Communicate_Met",
        "Communicate_Missing_Info",
        "Message_Count",
        "Tool_Calls_Count",
        "Agent_Cost_USD",
        "Results_JSON_Path",
    ]

    existing_keys = set()
    if csv_file.exists():
        try:
            with open(csv_file, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    existing_keys.add((row.get("Run_Folder"), row.get("Task_ID"), row.get("Trial")))
        except Exception:
            pass

    rows_to_write = []
    results_files = sorted(base.glob("*/results.json"), key=lambda p: p.parent.stat().st_mtime)

    for r_path in results_files:
        try:
            folder_name = r_path.parent.name
            results = Results.load(r_path)
            domain = getattr(results.info.environment_info, "domain_name", "") or folder_name.split("_")[2] if len(folder_name.split("_")) > 2 else "unknown"
            agent = getattr(results.info.agent_info, "name", "rac_planner")

            for sim in results.simulations:
                task_id = str(sim.task_id)
                trial = str(getattr(sim, "trial", 0))
                key = (folder_name, task_id, trial)

                if key in existing_keys:
                    continue

                r_info = sim.reward_info
                reward_val = getattr(r_info, "reward", 0.0) if r_info else 0.0

                db_match = False
                db_reward = 0.0
                if r_info and getattr(r_info, "db_check", None):
                    db_match = getattr(r_info.db_check, "db_match", False)
                    db_reward = getattr(r_info.db_check, "db_reward", 0.0)

                action_reward = 0.0
                if r_info and getattr(r_info, "action_checks", None):
                    actions = r_info.action_checks
                    matched = sum(1 for a in actions if getattr(a, "action_match", False))
                    action_reward = matched / max(1, len(actions))

                comm_met = True
                comm_missing = ""
                if r_info and getattr(r_info, "communicate_checks", None):
                    for cc in r_info.communicate_checks:
                        if not getattr(cc, "met", False):
                            comm_met = False
                            comm_missing = str(getattr(cc, "info", ""))

                msg_count = len(sim.messages) if sim.messages else 0
                tool_calls_count = sum(1 for m in sim.messages if getattr(m, "tool_calls", None))
                cost = getattr(sim, "agent_cost", 0.0) or 0.0

                t_stamp = getattr(sim, "start_time", None)
                if hasattr(t_stamp, "isoformat"):
                    t_str = t_stamp.isoformat()
                elif t_stamp:
                    t_str = str(t_stamp)
                else:
                    t_str = folder_name[:15]

                rows_to_write.append(
                    {
                        "Timestamp": t_str,
                        "Run_Folder": folder_name,
                        "Domain": domain.upper(),
                        "Task_ID": task_id,
                        "Trial": trial,
                        "Agent": agent,
                        "Reward": f"{reward_val:.4f}",
                        "DB_Match": str(db_match),
                        "DB_Reward": f"{db_reward:.4f}",
                        "Action_Reward": f"{action_reward:.4f}",
                        "Communicate_Met": str(comm_met),
                        "Communicate_Missing_Info": comm_missing,
                        "Message_Count": msg_count,
                        "Tool_Calls_Count": tool_calls_count,
                        "Agent_Cost_USD": f"{cost:.4f}",
                        "Results_JSON_Path": str(r_path.resolve()),
                    }
                )
        except Exception as e:
            print(f"Skipping {r_path}: {e}")

    if rows_to_write:
        write_header = not csv_file.exists() or csv_file.stat().st_size == 0
        with open(csv_file, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerows(rows_to_write)
        print(f"Appended {len(rows_to_write)} run entries to {csv_file}")
    else:
        print(f"CSV is up to date: {csv_file}")

    return csv_file


if __name__ == "__main__":
    update_run_history_csv()

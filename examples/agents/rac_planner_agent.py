#!/usr/bin/env python3
"""
RAC Planner Agent Evaluation Example for Tau2-Bench.

This script demonstrates evaluating an agent powered by RAC (React Agent Compensation)
planner (`react_agent_compensation.planner`) with multi-candidate generation,
Pareto front selection, and RecoveryManager failure recovery on Tau2 benchmark tasks.

Usage:
    python examples/agents/rac_planner_agent.py
"""

from typing import Optional
from loguru import logger

from tau2.agent.rac_planner import RACPlannerAgent, create_rac_planner_agent
from tau2.data_model.simulation import TextRunConfig
from tau2.registry import registry
from tau2.runner import get_tasks, run_single_task


if __name__ == "__main__":
    # Ensure rac_planner is registered in tau2
    if "rac_planner" not in registry.get_agents():
        registry.register_agent_factory(create_rac_planner_agent, "rac_planner")

    # Load tasks from mock domain
    tasks = get_tasks("mock", task_ids=["create_task_1"])
    task = tasks[0]

    print("=" * 60)
    print("Running RAC Planner Agent evaluation on Tau2 Mock task...")
    print(f"Task ID: {task.id}")
    print("=" * 60)

    config = TextRunConfig(
        domain="mock",
        agent="rac_planner",
        llm_agent="openai/gpt-5.6-luna",
        llm_user="openai/gpt-5.6-luna",
    )

    result = run_single_task(
        config=config,
        task=task,
        seed=42,
    )

    print()
    print("=" * 60)
    print("Evaluation Results")
    print("=" * 60)
    print(f"Reward: {result.reward_info.reward if result.reward_info else 'N/A'}")
    print(f"Total messages: {len(result.messages)}")

    print("\nConversation transcript:")
    for msg in result.messages:
        role = msg.role.value if hasattr(msg.role, "value") else msg.role
        if getattr(msg, "content", None):
            print(f"  [{role}] {str(msg.content)[:120]}")
        elif hasattr(msg, "tool_calls") and msg.tool_calls:
            names = [tc.name for tc in msg.tool_calls]
            print(f"  [{role}] Tool calls: {names}")
        else:
            print(f"  [{role}] (tool result)")

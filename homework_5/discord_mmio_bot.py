#!/usr/bin/env python3
"""Part 6 — Discord bot for MMIO (Homework 5).

Modes (env):
  HW5_DISCORD_MODE=faq          (default) — short answers about MMIO, no API solves.
  HW5_DISCORD_MODE=mmio_solve   — on @mention, runs one `test_agent.py` smoke (expensive).

Requires: pip install discord.py
  export DISCORD_TOKEN=...
  export OPENAI_API_KEY=...   # only for mmio_solve mode

Run from repository root:
  python homework_5/discord_mmio_bot.py
"""
from __future__ import annotations

import asyncio
import os
import subprocess
import sys
from pathlib import Path

_HW5 = Path(__file__).resolve().parent
_REPO = _HW5.parent

try:
    import discord
    from discord.ext import commands
except ImportError as e:
    print("Install discord.py: pip install discord.py", file=sys.stderr)
    raise SystemExit(1) from e


FAQ_TEXT = """**MMIO bot (homework)** — Multimodal Interactive Optimization for polling places.

Commands when you @mention me:
- Ask what MMIO does, archetypes (cluster, coverage_gap, contiguity, shape_niceness), or the 2×2 experiment (modality × vague/precise).
- Full MILP+agent solves are disabled unless the maintainer sets `HW5_DISCORD_MODE=mmio_solve` (uses OpenAI + Gurobi; costs money).

Repo: run `python test_agent.py --pair_dir full_dataset/contiguity/pairs/contiguity_med_00 --query_type vague --model gpt-5-mini` locally.
"""


def _maybe_run_mmio_solve() -> str:
    pair = os.environ.get(
        "HW5_SOLVE_PAIR_DIR",
        str(_REPO / "full_dataset/cluster/pairs/cluster_med_00"),
    )
    if not Path(pair).is_dir():
        return f"Configured pair dir missing: {pair}"
    cmd = [
        sys.executable,
        str(_REPO / "test_agent.py"),
        "--pair_dir",
        pair,
        "--query_type",
        os.environ.get("HW5_SOLVE_QUERY_TYPE", "vague"),
        "--model",
        os.environ.get("HW5_SOLVE_MODEL", "gpt-5-mini"),
        "--max_iters",
        os.environ.get("HW5_SOLVE_MAX_ITERS", "12"),
        "--no_visual",
    ]
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(_REPO),
            capture_output=True,
            text=True,
            timeout=int(os.environ.get("HW5_SOLVE_TIMEOUT_SEC", "600")),
        )
    except subprocess.TimeoutExpired:
        return "MMIO solve timed out."
    tail = (proc.stdout + "\n" + proc.stderr)[-3500:]
    if proc.returncode != 0:
        return f"Solve failed (code {proc.returncode}). Tail:\n```\n{tail}\n```"
    return f"Solve finished (stdout/stderr tail):\n```\n{tail}\n```"


def main() -> None:
    token = os.environ.get("DISCORD_TOKEN")
    if not token:
        print("Set DISCORD_TOKEN", file=sys.stderr)
        raise SystemExit(1)

    mode = os.environ.get("HW5_DISCORD_MODE", "faq").lower().strip()

    intents = discord.Intents.default()
    intents.message_content = True
    intents.members = True
    bot = commands.Bot(command_prefix="!", intents=intents)

    @bot.event
    async def on_ready():
        print(f"{bot.user} online (mode={mode}).")

    @bot.event
    async def on_message(message: discord.Message):
        if message.author.bot or not bot.user:
            return
        if not bot.user.mentioned_in(message):
            await bot.process_commands(message)
            return

        body = (
            message.content.replace(f"<@{bot.user.id}>", "")
            .replace(f"<@!{bot.user.id}>", "")
            .strip()
        )
        if not body:
            await message.channel.send(
                "Mention me with a question, e.g. what is the MMIO benchmark?"
            )
            await bot.process_commands(message)
            return

        async with message.channel.typing():
            if mode == "mmio_solve":
                out = await asyncio.to_thread(_maybe_run_mmio_solve)
            else:
                out = FAQ_TEXT + "\n\n_User message:_ " + body[:500]
            if len(out) > 1900:
                out = out[:1897] + "..."
            await message.channel.send(out)

        await bot.process_commands(message)

    bot.run(token)


if __name__ == "__main__":
    main()

"""Command line entry point.

    python -m ds_agent.cli            # web UI (default), browser opens automatically
    python -m ds_agent.cli serve      # same, with --port/--no-browser options
    python -m ds_agent.cli ask "..."  # one-shot question
    python -m ds_agent.cli chat       # terminal REPL (fallback)
    python -m ds_agent.cli build-data # (re)build the demo warehouse
"""

from __future__ import annotations

import argparse
import sys
import threading
import webbrowser
from pathlib import Path
from typing import Dict, List

from .agent import DSQueryAgent
from .config import settings
from .data import build_demo_warehouse

BOLD, DIM, CYAN, GREEN, RED, RESET = "\033[1m", "\033[2m", "\033[36m", "\033[32m", "\033[31m", "\033[0m"

EXAMPLES = [
    "近7天 GMV 为什么跌了？",
    "拆解一下各渠道的贡献",
    "近30天各渠道 GMV 排名",
    "看下这个 AB 实验的结果",
    "设计一个支付转化率实验，基线3%，希望检测2%的相对提升，每天5万流量",
    "用双重差分评估补贴政策对城市 GMV 的影响",
    "有哪些实验类型？",
    "把分析写成周报",
]


def _print_blocks(blocks: List[Dict]) -> None:
    """Render rich blocks in a terminal-friendly way."""
    for block in blocks:
        kind = block.get("type")
        if kind == "kpi":
            print()
            for item in block["items"]:
                sub = f"  {DIM}{item.get('sub', '')}{RESET}" if item.get("sub") else ""
                print(f"  {BOLD}{item['value']:<14}{RESET} {item['label']}{sub}")
        elif kind == "table":
            print(f"\n  {CYAN}{block['title']}{RESET}")
            print("  " + block["markdown"].replace("\n", "\n  "))
        elif kind == "chart":
            print(f"  {DIM}[chart: {block['title']}]{RESET}")


def _print_reply(reply) -> None:
    print()
    print(reply.text)
    _print_blocks(reply.blocks)
    if reply.artifacts:
        print()
        for path in reply.artifacts:
            print(f"  {GREEN}artifact{RESET} {path}")
    print()


def ensure_data(force: bool = False) -> None:
    """Create the demo warehouse on first use."""
    if force or not settings.warehouse.exists():
        print(f"{DIM}building demo warehouse at {settings.warehouse}{RESET}")
        build_demo_warehouse(settings.warehouse, force=force)


def cmd_chat(args: argparse.Namespace) -> None:
    ensure_data()
    agent = DSQueryAgent(warehouse=settings.warehouse)
    print(f"{BOLD}DS Agent{RESET} {DIM}- natural language data science. Type 'help' for capabilities, 'exit' to quit.{RESET}")
    try:
        while True:
            try:
                question = input(f"{CYAN}ds>{RESET} ").strip()
            except EOFError:
                break
            if question.lower() in ("exit", "quit", ":q"):
                break
            if not question:
                continue
            _print_reply(agent.ask(question))
    finally:
        agent.close()
        print(f"{DIM}bye{RESET}")


def cmd_ask(args: argparse.Namespace) -> None:
    ensure_data()
    agent = DSQueryAgent(warehouse=settings.warehouse)
    try:
        _print_reply(agent.ask(" ".join(args.question)))
    finally:
        agent.close()


def cmd_serve(args: argparse.Namespace) -> None:
    ensure_data()
    import uvicorn

    from .web import create_app

    app = create_app()
    url = f"http://{args.host}:{args.port}"
    if args.open_browser:
        # Open the browser once the server is accepting connections; a fixed
        # short delay is enough for uvicorn's startup on a local machine.
        threading.Timer(1.2, webbrowser.open, [url]).start()
    print(f"{BOLD}DS Agent web UI{RESET} -> {url}" + ("" if args.open_browser else f" {DIM}(browser not opened){RESET}"))
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


def cmd_build_data(args: argparse.Namespace) -> None:
    path, orders = build_demo_warehouse(settings.warehouse, force=True)
    print(f"{GREEN}warehouse ready{RESET}: {path} ({len(orders):,} daily rows)")


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ds-agent", description="Natural-language data science agent")
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("chat", help="interactive REPL").set_defaults(func=cmd_chat)

    ask_parser = sub.add_parser("ask", help="one-shot question")
    ask_parser.add_argument("question", nargs="+")
    ask_parser.set_defaults(func=cmd_ask)

    serve_parser = sub.add_parser("serve", help="run the web UI (default command)")
    serve_parser.add_argument("--host", default="127.0.0.1")
    serve_parser.add_argument("--port", type=int, default=8000)
    serve_parser.add_argument("--no-browser", dest="open_browser", action="store_false", help="do not auto-open the browser")
    serve_parser.set_defaults(func=cmd_serve, open_browser=True)

    sub.add_parser("build-data", help="rebuild the demo warehouse").set_defaults(func=cmd_build_data)

    args = parser.parse_args(argv)
    if not getattr(args, "command", None):
        # Default interaction is the web UI: richer than the terminal (charts,
        # tables, KPI cards render properly) and it needs no terminal skills.
        args.func = cmd_serve
        args.host, args.port, args.open_browser = "127.0.0.1", 8000, True
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())

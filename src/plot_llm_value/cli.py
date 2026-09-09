"""Agent-facing command boundary. Stdout is always a complete result."""

import argparse
import json
import math
import os
import sys
from importlib.resources import files
from pathlib import Path

from dotenv import dotenv_values

from .api import APIError, fetch_models
from .data import SelectionError, compare, normalize, select


class Parser(argparse.ArgumentParser):
    def error(self, message):
        raise SelectionError(message)


def _arguments(argv):
    command = argv[0] if argv and argv[0] in {"plot", "data", "models"} else "plot"
    if argv and argv[0] == command:
        argv = argv[1:]
    parser = Parser(add_help=False)
    selectors = parser.add_mutually_exclusive_group()
    for option in ("provider", "model", "variant"):
        selectors.add_argument("--" + option, nargs="+", action="extend")
    parser.add_argument(
        "--metric", choices=["intelligence", "coding", "agentic", "agent"], default="intelligence"
    )
    parser.add_argument("--linear-x", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--timeout", type=float, default=30)
    parser.add_argument("--env-file", type=Path)
    args = parser.parse_args(argv)
    if command != "plot" and (args.linear_x or args.output):
        raise SelectionError("--linear-x and --output are plot-only options.")
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        raise SelectionError("--timeout must be positive and finite.")
    if args.output and args.output.suffix.lower() not in {".png", ".svg", ".pdf"}:
        raise SelectionError("--output must end in .png, .svg, or .pdf.")
    return command, args


def _key(env_file):
    values = {}
    if env_file is not None:
        if not env_file.is_file():
            raise APIError("--env-file does not exist or is not a file.")
        values = dotenv_values(env_file, interpolate=False)
    return (
        os.environ.get("ARTIFICIAL_ANALYSIS_API_KEY")
        or os.environ.get("AA_API_KEY")
        or values.get("ARTIFICIAL_ANALYSIS_API_KEY")
        or values.get("AA_API_KEY")
    )


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--help" in argv or "-h" in argv or argv == ["help"]:
        print(files("plot_llm_value").joinpath("help.txt").read_text(), end="")
        return 0
    try:
        command, args = _arguments(argv)
        snapshot = fetch_models(_key(args.env_file), timeout=args.timeout)
        rows = select(
            normalize(snapshot["data"]),
            providers=args.provider,
            models=args.model,
            variants=args.variant,
        )
        if command == "models":
            result = {"schema_version": 1, "source": snapshot["source"], "rows": rows}
        else:
            result = compare(rows, args.metric)
            result["source"] = snapshot["source"]
            if command == "plot":
                from .plot import save_plot

                result = save_plot(result, args.output, linear_x=args.linear_x)
        print(json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2))
        return 0
    except SelectionError as error:
        print(f"plot-llm-value: {error}", file=sys.stderr)
        return 2
    except (APIError, OSError) as error:
        print(f"plot-llm-value: {error}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("plot-llm-value: interrupted", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

"""Command line entry point: ``ecocorridor`` or ``python -m ecocorridor``."""

from __future__ import annotations

import argparse

from . import __version__


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="ecocorridor",
        description="EcoCorridor: ecological corridors and landscape connectivity (web application).",
    )
    parser.add_argument("--host", default="127.0.0.1", help="address to listen on (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="port (default: 8000)")
    parser.add_argument("--version", action="version", version=f"ecocorridor {__version__}")
    args = parser.parse_args(argv)

    import uvicorn

    uvicorn.run("ecocorridor.api:app", host=args.host, port=args.port)


if __name__ == "__main__":
    main()

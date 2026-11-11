"""Command line entry point: ``ecocorridor`` or ``python -m ecocorridor``."""

from __future__ import annotations

import argparse
import sys
import threading
import webbrowser

from . import __version__


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="ecocorridor",
        description="EcoCorridor: ecological corridors and landscape connectivity (web application).",
    )
    parser.add_argument("--host", default="127.0.0.1", help="address to listen on (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="port (default: 8000)")
    parser.add_argument("--open", dest="open_browser", action="store_true", default=None,
                        help="open the application in the default browser once the server is up")
    parser.add_argument("--no-browser", dest="open_browser", action="store_false",
                        help="do not open a browser (default for the pip-installed command)")
    parser.add_argument("--version", action="version", version=f"ecocorridor {__version__}")
    args = parser.parse_args(argv)

    # The packaged Windows application (PyInstaller) opens the browser by default; the pip command does not.
    open_browser = getattr(sys, "frozen", False) if args.open_browser is None else args.open_browser

    import uvicorn

    from .api import app
    from .dataset import CACHE_DIR

    url = f"http://{'127.0.0.1' if args.host in ('0.0.0.0', '::') else args.host}:{args.port}/"
    print(f"EcoCorridor {__version__}: {url}")
    print(f"Кэш загруженных территорий: {CACHE_DIR}")
    print("Закройте это окно, чтобы остановить приложение.")
    if open_browser:
        threading.Timer(1.5, webbrowser.open, args=(url,)).start()
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()

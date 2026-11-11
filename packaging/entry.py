"""PyInstaller entry point for the Windows application."""

import multiprocessing

from ecocorridor.__main__ import main

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()

"""Main entry point for gwsim package."""

from __future__ import annotations

if __name__ == "__main__":
    from gwsim.utils.log import setup_logger

    setup_logger(print_version=True)

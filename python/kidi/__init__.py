"""Kidi's native command-line interface."""

import sys
from collections.abc import Sequence

from ._native import __version__, cli

__all__ = ["__version__", "main"]


def main(args: Sequence[str] | None = None) -> int:
    """Run the native CLI and return its exit code; args excludes the program name."""
    def resolve_model(reference: str, cache: str) -> str:
        from .hub import resolve

        return str(resolve(reference, cache))

    return cli(list(sys.argv[1:] if args is None else args), resolve_model)
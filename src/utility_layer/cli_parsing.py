"""Small CLI parsing helpers used across layers."""

from __future__ import annotations

import argparse


def str2bool(x: str) -> bool:
    """Turn common text values like yes/no into real booleans."""
    # Some argparse paths already hand over a real bool.
    if isinstance(x, bool):
        return x
    # This keeps the CLI a bit forgiving about capitalization.
    x = x.lower()
    if x in {"true", "1", "yes", "y"}:
        return True
    if x in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {x}")


def parse_int_list(values: list[str]) -> list[int]:
    """Parse CLI list input like ["1,2", "3"] into [1, 2, 3]."""
    result: list[int] = []
    for value in values:
        for part in value.split(","):
            part = part.strip()
            if part:
                result.append(int(part))
    return result


def parse_str_list(values: list[str]) -> list[str]:
    """Parse CLI list input like ["mlp,kan", "lan"] into ["mlp", "kan", "lan"]."""
    result: list[str] = []
    for value in values:
        for part in value.split(","):
            part = part.strip()
            if part:
                result.append(part)
    return result

"""Tiny helper to load .env from experiments/ regardless of cwd.

Why: every script does `os.environ['ANTHROPIC_API_KEY']` etc. Without dotenv
the user has to `set/export` them manually before each call. With this helper
they can just keep the .env file beside .env.example and forget about it.

Usage at the top of any script:
  from utils.env import load_env
  load_env()
"""

from __future__ import annotations

from pathlib import Path


def load_env() -> None:
    """Walk up from cwd looking for experiments/.env; load it if found."""
    try:
        from dotenv import load_dotenv  # type: ignore
    except ImportError:
        return  # silently skip if dotenv not installed

    here = Path.cwd()
    for parent in [here, *here.parents]:
        candidate = parent / ".env"
        if candidate.exists() and (parent / ".env.example").exists():
            load_dotenv(candidate)
            return
    # Fallback: also try the experiments dir relative to this file
    fallback = Path(__file__).resolve().parent.parent.parent / ".env"
    if fallback.exists():
        load_dotenv(fallback)

#!/usr/bin/env python3
"""Compatibility entry point for the renamed failure-instruction generator."""

from __future__ import annotations

from generate_failure_instructions import main

if __name__ == "__main__":
    raise SystemExit(main())

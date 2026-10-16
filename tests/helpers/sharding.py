"""Inspect compiled communication without depending on HLO instruction names."""

import re

_COLLECTIVE = re.compile(
    r"\b(all-reduce|all-gather|all-to-all|collective-permute|"
    r"reduce-scatter|collective-broadcast)(?:-start|-done)?\("
)


def compiled_collectives(hlo: str) -> set[str]:
    """Return the collective operations present in compiled HLO text."""
    return set(_COLLECTIVE.findall(hlo))

"""Graph collection: tier-parameterized factory for all graph types.

Usage:
    from hammy_lib.collection import TIERS, make_tier, make_graph

    graphs = make_tier("XS")           # dict name → Graph instance
    g = make_graph("SquareLattice", "S")
    all_graphs = {tier: make_tier(tier) for tier in TIERS}
"""

from __future__ import annotations
from typing import Callable
from .graph import (
    PathGraph, CycleGraph, SquareLattice, HexLattice,
    BrickLattice, CubicLattice, Torus2D, Torus3D, IcosahedralSphere, Graph,
)


# Tier definitions: name → constructor args per tier
# Each entry: (constructor, args_per_tier)
# args_per_tier is a dict tier_name → tuple of constructor args

_TIER_PARAMS: dict[str, dict[str, tuple]] = {
    "PathGraph": {
        "XS": (144,),
        "S": (576,),
        "M": (2304,),
        "L": (5184,),
    },
    "CycleGraph": {
        "XS": (144,),
        "S": (576,),
        "M": (2304,),
        "L": (5184,),
    },
    "SquareLattice": {
        "XS": (12, 12),
        "S": (24, 24),
        "M": (48, 48),
        "L": (72, 72),
    },
    "HexLattice": {
        "XS": (12, 12),
        "S": (24, 24),
        "M": (48, 48),
        "L": (72, 72),
    },
    "BrickLattice": {
        "XS": (12, 12),
        "S": (24, 24),
        "M": (48, 48),
        "L": (72, 72),
    },
    "CubicLattice": {
        "XS": (6, 6, 4),
        "S": (8, 8, 9),
        "M": (12, 12, 16),
        "L": (16, 18, 18),
    },
    "Torus2D": {
        "XS": (12, 12),
        "S": (24, 24),
        "M": (48, 48),
        "L": (72, 72),
    },
    "Torus3D": {
        "XS": (6, 6, 4),
        "S": (8, 8, 9),
        "M": (12, 12, 16),
        "L": (16, 18, 18),
    },
    # IcosahedralSphere: single arg k, N = 10*k²+2
    # XS~144: k=4→162, S~576: k=7→492, M~2304: k=15→2252, L~5184: k=23→5292
    "IcosahedralSphere": {
        "XS": (4,),
        "S": (7,),
        "M": (15,),
        "L": (23,),
    },
}

_CONSTRUCTORS: dict[str, type] = {
    "PathGraph": PathGraph,
    "CycleGraph": CycleGraph,
    "SquareLattice": SquareLattice,
    "HexLattice": HexLattice,
    "BrickLattice": BrickLattice,
    "CubicLattice": CubicLattice,
    "Torus2D": Torus2D,
    "Torus3D": Torus3D,
    "IcosahedralSphere": IcosahedralSphere,
}

TIERS = ("XS", "S", "M", "L")
GRAPH_TYPES = tuple(_TIER_PARAMS.keys())


def make_graph(graph_type: str, tier: str) -> Graph:
    """Construct a single graph of the given type and tier.

    Args:
        graph_type: one of GRAPH_TYPES.
        tier: one of TIERS ("XS", "S", "M", "L").

    Returns:
        Constructed (but not calculated) Graph instance.
    """
    if graph_type not in _CONSTRUCTORS:
        raise ValueError(f"Unknown graph type {graph_type!r}. Available: {GRAPH_TYPES}")
    if tier not in TIERS:
        raise ValueError(f"Unknown tier {tier!r}. Available: {TIERS}")
    cls = _CONSTRUCTORS[graph_type]
    args = _TIER_PARAMS[graph_type][tier]
    return cls(*args)


def make_tier(tier: str) -> dict[str, Graph]:
    """Construct all graph types for a given tier.

    Args:
        tier: one of TIERS ("XS", "S", "M", "L").

    Returns:
        Dict mapping graph type name → Graph instance.
    """
    return {name: make_graph(name, tier) for name in GRAPH_TYPES}


def tier_node_counts(tier: str) -> dict[str, int]:
    """Return expected node counts for each graph type in the tier."""
    result = {}
    for name in GRAPH_TYPES:
        args = _TIER_PARAMS[name][tier]
        n = 1
        for a in args:
            n *= a
        result[name] = n
    return result

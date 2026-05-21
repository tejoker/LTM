"""Shared scatter primitives, re-exported for baselines that don't import
ltm.models.rlic_layer to keep the dependency graph clean."""

from ..models.rlic_layer import _segment_softmax, _segment_sum  # noqa: F401

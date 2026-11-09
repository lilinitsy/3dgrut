# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Which LOD slots a frame needs for a given cone cap, and what each is inflated by.

Slot 0 is the uninflated main BVH (band [0, s_0)), slot 1 the originals (band [s_0, s_1)), and slot 1 + k cut k
(band [s_k, s_{k+1})). Each slot's proxies are inflated by its band's upper edge, except the top enabled slot, whose
band is open above: no cone is wider than the cap, so it is inflated by the cap. The kernel does not clamp band
selection by the cap, so the engine builds and enables only the slots whose band starts below it.
"""

from typing import List, Sequence, Tuple

from threedgrut_playground.tracer import MAX_LOD_SLOTS


def plan_lod_slots(
    level_sigmas: Sequence[float], thin_sigma: float, max_sigma: float, max_slots: int = MAX_LOD_SLOTS
) -> Tuple[int, float]:
    """``(num_slots, effective_max_sigma)`` for ``level_sigmas`` = [s_1, ..., s_{K+1}].

    The cap is clamped to the upper band edge of the highest slot that can exist (s_{K+1} unless ``max_slots`` cuts
    the layout short), so no slot needs proxies wider than its band. ``num_slots`` counts slot 0 and covers the
    longest prefix of slots 1, 2, ... whose band lower edge (s_0, s_1, ..., s_K) is below the clamped cap. Slot
    S - 1's upper band edge (the next slot's lower edge, or the top) is then at least the cap.
    """
    num_levels = len(level_sigmas) - 1
    top_slot = min(1 + num_levels, max_slots - 1)
    cap = min(float(max_sigma), float(level_sigmas[top_slot - 1]))
    lower = [float(thin_sigma)] + [float(s) for s in level_sigmas[:num_levels]]  # slots 1..1+K
    num_slots = 1
    for slot in range(1, top_slot + 1):
        if not lower[slot - 1] < cap:
            break
        num_slots = slot + 1
    return num_slots, cap


def lod_slot_inflations(level_sigmas: Sequence[float], num_slots: int, cap: float) -> List[float]:
    """The inflations of slots 1..``num_slots`` - 1 for the clamped ``cap`` from ``plan_lod_slots``: each slot's upper
    band edge (s_1, ..., s_{S-1}), except that none exceeds the cap. Only the top slot's band reaches the cap, so it
    alone is inflated by the cap; the kernel never widens beyond it (``ConeParams.maxSigma``)."""
    return [min(float(s), float(cap)) for s in level_sigmas[: num_slots - 1]]

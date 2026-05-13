from __future__ import annotations

from itinerary_models import Itinerary

from .models import ItinerarySkeleton
from .renderer import render_itinerary_deterministically


def write_final_itinerary(*, skeleton: ItinerarySkeleton) -> Itinerary:
    """Render the validated skeleton into the existing Itinerary schema.

    This first integration keeps final writing deterministic. A future LLM
    writer can be inserted here as long as it is followed by the validator and
    falls back to this renderer on hard failures.
    """
    return render_itinerary_deterministically(skeleton)


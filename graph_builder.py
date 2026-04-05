"""
Graph builder — data preparation layer.

Reads attraction meta-data from JSONL files, parses addresses, computes
NEAR pairs via Haversine distance, and returns structured data ready for
mem0 to persist into Neo4j.

All actual graph database operations are delegated to the mem0 Memory
instance (via ``memory.build_attraction_graph``,
``memory.match_attraction_categories``, ``memory.query_candidate_attractions``).
"""

import json
import re
import logging
from math import asin, cos, radians, sin, sqrt
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)

NEAR_THRESHOLD_KM = 2.0

_LAT_BUCKET_SIZE = 0.02  # ~2 km in latitude degrees

# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def haversine_distance_km(
    lat1: float, lon1: float, lat2: float, lon2: float, radius_km: float = 6371.0
) -> float:
    d_lat = radians(lat2 - lat1)
    d_lon = radians(lon2 - lon1)
    a = sin(d_lat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(d_lon / 2) ** 2
    return radius_km * 2 * asin(sqrt(a))


def _lat_bucket(lat: float) -> int:
    return int(lat / _LAT_BUCKET_SIZE)


# ---------------------------------------------------------------------------
# Address / city parsing
# ---------------------------------------------------------------------------

_CITY_RE = re.compile(
    r",\s*([A-Za-z\s\.\'-]+?)\s*,\s*[A-Z]{2}\s+\d{5}"
)

def parse_city_from_address(address: str) -> Optional[str]:
    """Extract the city name from a US-style address string."""
    if not address:
        return None
    m = _CITY_RE.search(address)
    if m:
        return m.group(1).strip()
    parts = [p.strip() for p in address.split(",")]
    if len(parts) >= 3:
        candidate = parts[-2].strip()
        if candidate and not re.fullmatch(r"[A-Z]{2}\s*\d*", candidate):
            return candidate
    return None


def state_name_from_filename(filename: str) -> str:
    """meta-Alaska-tourist.json  ->  Alaska"""
    stem = Path(filename).stem
    if stem.startswith("meta-") and stem.endswith("-tourist"):
        return stem[5:-8].replace("_", " ")
    return stem


# ---------------------------------------------------------------------------
# Build available-states index (lightweight: just scan filenames)
# ---------------------------------------------------------------------------

def build_available_states(meta_dir: Union[str, Path]) -> Dict[str, str]:
    """Return {normalised_state_name: actual_state_name} from filenames.

    Keys are lower-cased for case-insensitive lookup; values are the
    canonical names used in filenames (e.g. "New York").
    """
    meta_dir = Path(meta_dir)
    index: Dict[str, str] = {}
    for fp in sorted(meta_dir.glob("meta-*-tourist.json")):
        state = state_name_from_filename(fp.name)
        index[state.lower()] = state
    logger.info("Available states index built: %d states", len(index))
    return index


# ---------------------------------------------------------------------------
# Build city → state index (for resolving city-only destinations)
# ---------------------------------------------------------------------------

def build_city_to_state_index(meta_dir: Union[str, Path]) -> Dict[str, str]:
    """Return {city_name: state_name} by scanning every meta-*-tourist.json."""
    meta_dir = Path(meta_dir)
    index: Dict[str, str] = {}
    for fp in sorted(meta_dir.glob("meta-*-tourist.json")):
        state = state_name_from_filename(fp.name)
        with open(fp, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                city = parse_city_from_address(row.get("address", ""))
                if city and city not in index:
                    index[city] = state
    logger.info("City→State index built: %d cities across meta files", len(index))
    return index


# ---------------------------------------------------------------------------
# Load & prepare attraction data (pure data, no graph DB calls)
# ---------------------------------------------------------------------------

def _load_meta_attractions(meta_dir: Union[str, Path], state: str) -> List[Dict]:
    """Read all attraction records from the meta file for *state*."""
    meta_dir = Path(meta_dir)
    fname = f"meta-{state.replace(' ', '_')}-tourist.json"
    fp = meta_dir / fname
    if not fp.exists():
        logger.warning("Meta file not found: %s", fp)
        return []
    attractions = []
    with open(fp, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            city = parse_city_from_address(row.get("address", ""))
            categories = [c for c in row.get("category", []) if c != "Tourist attraction"]
            attractions.append({
                "name": row["name"],
                "latitude": row.get("latitude"),
                "longitude": row.get("longitude"),
                "avg_rating": row.get("avg_rating"),
                "num_of_reviews": row.get("num_of_reviews"),
                "address": row.get("address", ""),
                "city": city,
                "categories": categories,
                # NOTE: the current meta JSON has no "description" field;
                # this will be "" until meta files are updated.
                "description": row.get("description", ""),
            })
    return attractions


def _compute_near_pairs(attractions: List[Dict]) -> List[Tuple[str, str]]:
    """Return (name_a, name_b) pairs within NEAR_THRESHOLD_KM using
    spatial bucketing for efficiency."""
    # Include attractions without a parseable city so NEAR edges still apply
    # (those nodes connect to State only in the graph).
    valid = [a for a in attractions if a.get("latitude") and a.get("longitude")]
    buckets: Dict[int, List[int]] = defaultdict(list)
    for idx, a in enumerate(valid):
        buckets[_lat_bucket(a["latitude"])].append(idx)

    near_pairs: List[Tuple[str, str]] = []
    for bucket_key, indices in buckets.items():
        neighbors = indices[:]
        for adj_key in (bucket_key - 1, bucket_key + 1):
            neighbors.extend(buckets.get(adj_key, []))

        for i, idx_a in enumerate(indices):
            for idx_b in neighbors:
                if idx_a >= idx_b:
                    continue
                a1, a2 = valid[idx_a], valid[idx_b]
                dist = haversine_distance_km(
                    a1["latitude"], a1["longitude"],
                    a2["latitude"], a2["longitude"],
                )
                if dist <= NEAR_THRESHOLD_KM:
                    near_pairs.append((a1["name"], a2["name"]))

    return near_pairs


def prepare_state_graph_data(
    state: str, meta_dir: Union[str, Path]
) -> dict:
    """Parse meta-data for *state* and return a dict ready for
    ``memory.build_attraction_graph()``.

    Returns
    -------
    dict with keys:
        state       – state name
        attractions – list of attraction dicts
        near_pairs  – list of (name_a, name_b) tuples
    """
    attractions = _load_meta_attractions(meta_dir, state)
    near_pairs = _compute_near_pairs(attractions) if attractions else []

    logger.info(
        "prepare_state_graph_data – state=%s  attractions=%d  near_pairs=%d",
        state, len(attractions), len(near_pairs),
    )
    return {
        "state": state,
        "attractions": attractions,
        "near_pairs": near_pairs,
    }

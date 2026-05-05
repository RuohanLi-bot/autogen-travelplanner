"""
Extract user preferences from reviews.jsonl and batch-load into mem0.

Reviews are read in file order, split into at most 10 equally-sized batches (sizes
differ by at most one when there are ≥10 reviews), then each batch is merged into
one text block before calling mem0.add().

Can be used standalone:
    python preload_memories.py

Or imported:
    from preload_memories import preload_reviews
    preload_reviews(memory, reviews_path, user_id)
"""

import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Union

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent
ENV_FILE = PROJECT_ROOT / ".env"
load_dotenv(ENV_FILE)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_REVIEWS_PATH = Path(
    os.environ.get(
        "REVIEWS_PATH",
        "/data/lrh/InteRecAgent/resources/google/output/User/"
        "1_user_110127197526819446448/reviews.jsonl",
    )
)
DEFAULT_USER_ID = os.environ.get("MEM0_USER_ID", "bryce_caster")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clean_categories(cats: List[str]) -> Tuple[str, ...]:
    """Remove 'Tourist attraction' and return a sorted, deduplicated tuple."""
    return tuple(sorted(set(c for c in cats if c != "Tourist attraction")))


def _even_chunk_ranges(n: int, k: int) -> List[Tuple[int, int]]:
    """Split ``range(n)`` into *k* contiguous slices with sizes differing by at most 1."""
    if n <= 0 or k <= 0:
        return []
    k = min(k, n)
    base, rem = divmod(n, k)
    ranges: List[Tuple[int, int]] = []
    start = 0
    for i in range(k):
        size = base + (1 if i < rem else 0)
        end = start + size
        ranges.append((start, end))
        start = end
    return ranges


def _build_batch_text(
    reviews: List[Dict], batch_index: int, num_batches: int
) -> str:
    """Merge one batch of reviews into a single preference statement."""
    header = (
        f"Travel experiences (batch {batch_index}/{num_batches}, "
        f"{len(reviews)} place(s)):\n"
    )
    items = []
    for r in reviews:
        rating = r.get("rating", "?")
        name = r.get("place_name", "unknown")
        city = r.get("city", "Unknown")
        cats = _clean_categories(r.get("category", []))
        cat_label = "/".join(cats) if cats else "general"
        text = (r.get("text") or "")[:150]
        items.append(f"- [{city} | {cat_label}] {name} ({rating}/5): {text}")
    return header + "\n".join(items)


# ---------------------------------------------------------------------------
# Core function (importable)
# ---------------------------------------------------------------------------

def preload_reviews(
    memory,
    reviews_path: Union[str, Path] = DEFAULT_REVIEWS_PATH,
    user_id: str = DEFAULT_USER_ID,
) -> int:
    """Load reviews from JSONL and add to mem0 in at most 10 evenly-sized batches.

    Batch count is ``min(10, n)`` where *n* is the total number of reviews (when
    ``n < 10``, each batch holds one review so we do not emit empty adds).
    Sizes differ by at most one when ``n >= 10``.

    Parameters
    ----------
    memory : mem0.Memory
        An already-initialised Memory instance.
    reviews_path : str | Path
        Path to reviews.jsonl.
    user_id : str
        mem0 user identifier.

    Returns
    -------
    int
        Number of batches added (``memory.add`` calls).
    """
    reviews_path = Path(reviews_path)
    logger.info("Loading reviews from %s ...", reviews_path)

    reviews: List[Dict] = []
    with open(reviews_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                reviews.append(json.loads(line))
    n = len(reviews)
    logger.info("  Total reviews: %d", n)
    TARGET_BATCHES = 10
    num_batches = min(TARGET_BATCHES, n)
    ranges = _even_chunk_ranges(n, num_batches)
    print(
        f"[preload] 共 {num_batches} 批、{n} 条 review；每批会调用 mem0.add（向量 + Neo4j 并行），"
        f"远程 Neo4j 或首次 embedding 时可能较慢，并非死机。",
        file=sys.stderr,
        flush=True,
    )
    total_batches = 0
    for bi, (start, end) in enumerate(ranges, start=1):
        chunk = reviews[start:end]
        text = _build_batch_text(chunk, bi, num_batches)
        print(
            f"[preload] 正在写入 batch {bi}/{num_batches}（reviews [{start}:{end}]，{len(chunk)} 条）…",
            file=sys.stderr,
            flush=True,
        )
        memory.add(text, user_id=user_id)
        total_batches += 1
        logger.info(
            "  Batch %d/%d: reviews [%d:%d] (%d items)",
            bi, num_batches, start, end, len(chunk),
        )

    logger.info("Done. %d batches loaded into mem0.", total_batches)
    return total_batches


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------

def main():
    from mem0 import Memory

    mem0_config = {
        "llm": {
            "provider": "openai",
            "config": {
                "model": "gpt-4o-mini",
                "api_key": os.environ.get("OPENAI_API_KEY"),
                "openai_base_url": os.environ.get("OPENAI_BASE_URL"),
                "temperature": 0.2,
                "max_tokens": 5000,
                "top_p": 1.0,
            },
        },
        "embedder": {
            "provider": "huggingface",
            "config": {"model": "sentence-transformers/all-MiniLM-L6-v2"},
        },
        "vector_store": {
            "provider": "qdrant",
            "config": {
                "path": os.path.join(
                    os.environ.get("MEM0_DIR", os.path.expanduser("~/.mem0")),
                    "vector_store",
                ),
                "on_disk": True,
                "embedding_model_dims": 384,
            },
        },
        "graph_store": {
            "provider": "neo4j",
            "config": {
                "url": os.environ.get("NEO4J_URL"),
                "username": os.environ.get("NEO4J_USERNAME"),
                "password": os.environ.get("NEO4J_PASSWORD"),
                "database": os.environ.get("NEO4J_DATABASE"),
            },
        },
    }

    print("Initializing mem0 ...")
    memory = Memory.from_config(mem0_config)
    n = preload_reviews(memory)
    print(f"Done. {n} batches loaded.")


if __name__ == "__main__":
    main()

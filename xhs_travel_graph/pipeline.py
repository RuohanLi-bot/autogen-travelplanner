from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from .cluster import XHSPlayModeClusterer
from .extractor import XHSTravelFactExtractor
from .graph_repository import Mem0Neo4jQueryRunner
from .graph_writer import XHSTravelGraphWriter
from .post_parser import load_autoglm_posts

logger = logging.getLogger(__name__)

try:
    from poi_research.llm_client import OpenAILLMClient
except Exception:  # pragma: no cover - normal path in this workspace uses poi_research.
    from openai import OpenAI

    class OpenAILLMClient:
        def __init__(
            self,
            model: str = "gpt-4o-mini",
            api_key: Optional[str] = None,
            base_url: Optional[str] = None,
        ):
            self.model = model
            self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
            self.base_url = base_url or os.environ.get("OPENAI_BASE_URL")
            self.client = OpenAI(api_key=self.api_key, base_url=self.base_url) if self.api_key else None

        def available(self) -> bool:
            return self.client is not None

        def generate_json(
            self,
            *,
            system_prompt: str,
            user_prompt: str,
            temperature: float = 0.2,
            default: Optional[Dict[str, Any]] = None,
        ) -> Dict[str, Any]:
            if self.client is None:
                return default or {}
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    temperature=temperature,
                    response_format={"type": "json_object"},
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                )
                return json.loads(response.choices[0].message.content or "{}")
            except Exception as exc:
                logger.warning("LLM JSON generation failed: %s", exc)
                return default or {}


def ingest_autoglm_json_to_structured_xhs_graph(
    *,
    json_path: Path,
    mem0_client: Any,
    run_id: str = "xhs",
    destination: str = "",
    write_schema: bool = True,
    cluster_play_modes: bool = True,
    extraction_limit: Optional[int] = None,
    llm_client: Optional[Any] = None,
) -> Dict[str, Any]:
    posts = load_autoglm_posts(Path(json_path), run_id=run_id)
    if extraction_limit is not None:
        posts = posts[:extraction_limit]

    extractor = XHSTravelFactExtractor(llm_client or OpenAILLMClient())
    facts_by_post = {}
    for post in posts:
        facts_by_post[post.post_id] = extractor.extract(post)

    runner = Mem0Neo4jQueryRunner(mem0_client)
    writer = XHSTravelGraphWriter(runner)
    if write_schema:
        writer.ensure_schema()
    writer.write_many(posts, facts_by_post)

    cluster_summary: Dict[str, Any] = {}
    if cluster_play_modes:
        cluster_summary = XHSPlayModeClusterer(runner).cluster_and_write(
            run_id=run_id,
            destination=destination,
        )

    return {
        "posts": len(posts),
        "route_variants": sum(len(v) for v in facts_by_post.values()),
        "play_modes": cluster_summary.get("play_modes", 0),
        "cluster_summary": cluster_summary,
    }


def dry_run_autoglm_json(json_path: Path, limit: int = 3, use_llm: bool = False) -> Dict[str, Any]:
    posts = load_autoglm_posts(Path(json_path), run_id="xhs")
    extractor = XHSTravelFactExtractor(OpenAILLMClient() if use_llm else None)
    samples: List[Dict[str, Any]] = []
    for post in posts[:limit]:
        facts = extractor.extract(post)
        samples.append(
            {
                "post_id": post.post_id,
                "title": post.title,
                "parse_quality": post.parse_quality,
                "route_variants": [_model_to_dict(fact) for fact in facts],
            }
        )
    return {"posts": len(posts), "sample_facts": samples}


def _model_to_dict(model: Any) -> Dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()

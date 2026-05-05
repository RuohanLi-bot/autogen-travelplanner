from __future__ import annotations

import logging
import os
import threading
import time
from pathlib import Path
from typing import Dict, List

import requests
from dotenv import load_dotenv
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .models import ResearchDoc
from runtime_stats import runtime_stats

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = PROJECT_ROOT / ".env"
load_dotenv(ENV_FILE)


class SerpSearchClient:
    def __init__(self, api_key: str | None = None, max_results: int = 4, min_interval_s: float = 0.25):
        self.api_key = api_key or os.environ.get("SERPAPI_API_KEY")
        self.max_results = max_results
        self.min_interval_s = min_interval_s
        self._query_cache: Dict[str, List[ResearchDoc]] = {}
        self._url_cache: Dict[str, ResearchDoc] = {}
        self._lock = threading.Lock()
        self._last_call_ts = 0.0
        self._disabled = False
        self._disabled_reason = ""
        self._session = requests.Session()
        self._session.trust_env = True
        retries = Retry(
            total=3,
            connect=3,
            read=3,
            backoff_factor=0.8,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=frozenset(["GET"]),
        )
        adapter = HTTPAdapter(max_retries=retries)
        self._session.mount("https://", adapter)
        self._session.mount("http://", adapter)

    def available(self) -> bool:
        return bool(self.api_key)

    def _respect_rate_limit(self) -> None:
        with self._lock:
            delta = time.time() - self._last_call_ts
            if delta < self.min_interval_s:
                time.sleep(self.min_interval_s - delta)
            self._last_call_ts = time.time()

    def search(self, query: str) -> List[ResearchDoc]:
        if query in self._query_cache:
            return self._query_cache[query]
        if not self.api_key:
            logger.warning("SerpApi key missing; skipping search for query=%s", query)
            self._query_cache[query] = []
            return []
        if self._disabled:
            logger.debug(
                "SerpApi disabled (%s); skipping query=%s",
                self._disabled_reason or "unspecified reason",
                query,
            )
            self._query_cache[query] = []
            return []

        self._respect_rate_limit()
        try:
            response = self._session.get(
                "https://serpapi.com/search.json",
                params={
                    "engine": "google",
                    "q": query,
                    "api_key": self.api_key,
                    "num": self.max_results,
                    "output": "json",
                },
                headers={"Accept": "application/json"},
                timeout=30,
            )
            response.raise_for_status()
            data = response.json()
            runtime_stats.record_search_call(success=True)
        except requests.HTTPError as exc:
            runtime_stats.record_search_call(success=False)
            status_code = exc.response.status_code if exc.response is not None else None
            if status_code in (401, 403):
                self._disabled = True
                self._disabled_reason = f"auth failure ({status_code})"
                logger.warning(
                    "SerpApi disabled due to authentication failure for query=%s: %s",
                    query,
                    exc,
                )
            elif status_code == 429:
                logger.warning("SerpApi rate limited for query=%s: %s", query, exc)
            else:
                logger.warning("SerpApi HTTP failure for query=%s: %s", query, exc)
            self._query_cache[query] = []
            return []
        except requests.RequestException as exc:
            runtime_stats.record_search_call(success=False)
            logger.warning("SerpApi transient network failure for query=%s: %s", query, exc)
            self._query_cache[query] = []
            return []

        error_message = str(data.get("error") or "").strip()
        if error_message:
            runtime_stats.record_search_call(success=False)
            lower_error = error_message.casefold()
            if "searches left" in lower_error or "plan" in lower_error or "limit" in lower_error:
                self._disabled = True
                self._disabled_reason = error_message
            logger.warning("SerpApi logical failure for query=%s: %s", query, error_message)
            self._query_cache[query] = []
            return []

        docs: List[ResearchDoc] = []

        answer_box = data.get("answer_box") or {}
        if isinstance(answer_box, dict):
            answer_url = answer_box.get("link") or answer_box.get("source")
            answer_text = " ".join(
                str(answer_box.get(key) or "")
                for key in ("title", "answer", "snippet", "snippet_highlighted_words")
            ).strip()
            if answer_url and answer_text:
                doc = ResearchDoc(
                    url=str(answer_url),
                    title=str(answer_box.get("title") or ""),
                    content=answer_text,
                    score=1.0,
                )
                self._url_cache[doc.url] = doc
                docs.append(doc)

        for idx, item in enumerate(data.get("organic_results", []), 1):
            url = item.get("link")
            if not url:
                continue
            if url in self._url_cache:
                docs.append(self._url_cache[url])
                continue
            snippet_parts = []
            for key in ("snippet", "snippet_highlighted_words"):
                value = item.get(key)
                if isinstance(value, list):
                    snippet_parts.extend(str(part) for part in value if part)
                elif value:
                    snippet_parts.append(str(value))
            rich_snippet = item.get("rich_snippet") or {}
            if isinstance(rich_snippet, dict):
                top = rich_snippet.get("top") or {}
                if isinstance(top, dict):
                    for value in top.values():
                        if isinstance(value, list):
                            snippet_parts.extend(str(part) for part in value if part)
                        elif value:
                            snippet_parts.append(str(value))
            doc = ResearchDoc(
                url=str(url),
                title=str(item.get("title") or ""),
                content=" ".join(snippet_parts).strip(),
                score=max(0.0, 1.0 - (idx - 1) * 0.1),
            )
            self._url_cache[doc.url] = doc
            docs.append(doc)
            if len(docs) >= self.max_results:
                break

        deduped_docs = []
        seen_urls = set()
        for doc in docs:
            if doc.url in seen_urls:
                continue
            seen_urls.add(doc.url)
            deduped_docs.append(doc)
        self._query_cache[query] = deduped_docs
        return deduped_docs

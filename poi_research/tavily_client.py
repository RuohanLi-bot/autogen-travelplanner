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


class TavilySearchClient:
    def __init__(self, api_key: str | None = None, max_results: int = 4, min_interval_s: float = 0.25):
        self.api_key = api_key or os.environ.get("TAVILY_API_KEY")
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
            allowed_methods=frozenset(["POST"]),
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
            logger.warning("Tavily API key missing; skipping search for query=%s", query)
            self._query_cache[query] = []
            return []
        if self._disabled:
            logger.debug(
                "Tavily disabled (%s); skipping query=%s",
                self._disabled_reason or "unspecified reason",
                query,
            )
            self._query_cache[query] = []
            return []

        self._respect_rate_limit()
        try:
            response = self._session.post(
                "https://api.tavily.com/search",
                json={
                    "api_key": self.api_key,
                    "query": query,
                    "search_depth": "advanced",
                    "include_raw_content": True,
                    "max_results": self.max_results,
                },
                headers={"Accept": "application/json"},
                timeout=30,
            )
            response.raise_for_status()
            data = response.json()
            runtime_stats.record_tavily_call(success=True)
        except requests.HTTPError as exc:
            runtime_stats.record_tavily_call(success=False)
            status_code = exc.response.status_code if exc.response is not None else None
            if status_code in (401, 403):
                self._disabled = True
                self._disabled_reason = f"auth failure ({status_code})"
                logger.warning(
                    "Tavily search disabled due to authentication failure for query=%s: %s",
                    query,
                    exc,
                )
            elif status_code == 432:
                self._disabled = True
                self._disabled_reason = "usage limit reached (432)"
                logger.warning(
                    "Tavily search disabled because the API usage limit was reached for query=%s: %s",
                    query,
                    exc,
                )
            else:
                logger.warning("Tavily search HTTP failure for query=%s: %s", query, exc)
            self._query_cache[query] = []
            return []
        except requests.RequestException as exc:
            runtime_stats.record_tavily_call(success=False)
            logger.warning(
                "Tavily transient network failure for query=%s: %s",
                query,
                exc,
            )
            self._query_cache[query] = []
            return []

        docs: List[ResearchDoc] = []
        for item in data.get("results", []):
            url = item.get("url")
            if not url:
                continue
            if url in self._url_cache:
                docs.append(self._url_cache[url])
                continue
            doc = ResearchDoc(
                url=url,
                title=item.get("title", "") or "",
                content=item.get("raw_content") or item.get("content") or "",
                score=float(item.get("score") or 0.0),
            )
            self._url_cache[url] = doc
            docs.append(doc)
        self._query_cache[query] = docs
        return docs

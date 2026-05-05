from __future__ import annotations

import logging
import threading
from typing import Any, Dict

logger = logging.getLogger(__name__)


class RuntimeStats:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.reset()

    def reset(self) -> None:
        with self._lock:
            self.search_api_call_count = 0
            self.search_api_failure_count = 0
            self.llm_call_count = 0
            self.llm_failure_count = 0
            self.prompt_tokens = 0
            self.completion_tokens = 0
            self.total_tokens = 0

    def record_search_call(self, *, success: bool) -> None:
        with self._lock:
            self.search_api_call_count += 1
            if not success:
                self.search_api_failure_count += 1

    def record_llm_call(
        self,
        *,
        success: bool,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        total_tokens: int = 0,
    ) -> None:
        with self._lock:
            self.llm_call_count += 1
            if not success:
                self.llm_failure_count += 1
            self.prompt_tokens += int(prompt_tokens or 0)
            self.completion_tokens += int(completion_tokens or 0)
            self.total_tokens += int(total_tokens or 0)

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "search_api_call_count": self.search_api_call_count,
                "search_api_failure_count": self.search_api_failure_count,
                "llm_call_count": self.llm_call_count,
                "llm_failure_count": self.llm_failure_count,
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
                "total_tokens": self.total_tokens,
            }

    def format_summary(self) -> str:
        stats = self.snapshot()
        return (
            "\n[run stats]\n"
            f"SerpApi calls: {stats['search_api_call_count']} "
            f"(failures: {stats['search_api_failure_count']})\n"
            f"LLM calls: {stats['llm_call_count']} "
            f"(failures: {stats['llm_failure_count']})\n"
            f"Token usage: prompt={stats['prompt_tokens']}, "
            f"completion={stats['completion_tokens']}, total={stats['total_tokens']}\n"
        )


runtime_stats = RuntimeStats()

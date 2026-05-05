from __future__ import annotations

from typing import Any, Dict, List, Optional, Protocol


class QueryRunner(Protocol):
    def query(self, cypher: str, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        ...


class Mem0Neo4jQueryRunner:
    def __init__(self, mem0_client: Any):
        graph_memory = getattr(mem0_client, "graph", None)
        raw_graph = getattr(graph_memory, "graph", None) or graph_memory
        if raw_graph is None or not hasattr(raw_graph, "query"):
            raise ValueError("mem0_client does not expose a Neo4j query runner; create it with graph enabled")
        self._graph = raw_graph

    def query(self, cypher: str, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        return self._graph.query(cypher, params=params or {})


class RecordingQueryRunner:
    def __init__(self):
        self.calls: List[Dict[str, Any]] = []

    def query(self, cypher: str, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        self.calls.append({"cypher": cypher, "params": params or {}})
        return []

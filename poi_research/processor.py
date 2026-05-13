from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Iterable, List, Optional

from .aggregator import Aggregator
from .extractor import EvidenceExtractor
from .factor_planner import FactorPlanner
from .llm_client import OpenAILLMClient
from .models import NormalizedPOI, POIStructuredInfo
from .normalizer import POINormalizer
from .query_generator import QueryGenerator
from .reflect import ReflectAndRefine
from .result_builder import ResultBuilder
from .serp_client import SerpSearchClient

logger = logging.getLogger(__name__)


class POIResearchPipeline:
    def __init__(self, llm_client: Optional[OpenAILLMClient] = None, serp_client: Optional[SerpSearchClient] = None):
        self.llm_client = llm_client or OpenAILLMClient()
        self.search_client = serp_client or SerpSearchClient()
        self.normalizer = POINormalizer()
        self.factor_planner = FactorPlanner(self.llm_client)
        self.query_generator = QueryGenerator(self.llm_client)
        self.extractor = EvidenceExtractor(self.llm_client)
        self.reflector = ReflectAndRefine(self.llm_client)
        self.aggregator = Aggregator()
        self.result_builder = ResultBuilder()

    def _research_one(self, poi: NormalizedPOI, debug: bool = False) -> POIStructuredInfo:
        factor_plan = self.factor_planner.plan(poi)
        queries = self.query_generator.generate(poi, factor_plan)
        if debug:
            logger.info("POI=%s queries=%s", poi.poi_id, queries)
        docs = []
        for query in queries:
            docs.extend(self.search_client.search(query))
        deduped_docs = {doc.url: doc for doc in docs}.values()
        extracted = self.extractor.extract(poi, deduped_docs)
        decision = self.reflector.decide(poi, extracted)
        if decision.needs_secondary_search:
            if debug:
                logger.info("POI=%s secondary_search=%s", poi.poi_id, decision.additional_queries)
            secondary_docs = []
            for query in decision.additional_queries:
                secondary_docs.extend(self.search_client.search(query))
            merged_docs = {doc.url: doc for doc in list(deduped_docs) + secondary_docs}.values()
            extracted = self.extractor.extract(poi, merged_docs)
        info = self.aggregator.aggregate(poi, extracted)
        return self.result_builder.build(info)

    def process_pois(self, poi_list: Iterable[dict], max_workers: int = 4, debug: bool = False) -> List[POIStructuredInfo]:
        normalized = self.normalizer.normalize(poi_list)
        if not normalized:
            return []
        results: List[POIStructuredInfo] = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self._research_one, poi, debug): poi.poi_id for poi in normalized
            }
            for future in as_completed(futures):
                poi_id = futures[future]
                try:
                    results.append(future.result())
                except Exception as exc:
                    logger.exception("POI research failed for %s: %s", poi_id, exc)
                    results.append(
                        POIStructuredInfo(
                            poi_id=poi_id,
                            place_name=poi_id,
                            category="unknown",
                            unresolved_questions=["Research pipeline failed"],
                            notes=[str(exc)],
                        )
                    )
        return sorted(results, key=lambda item: item.poi_id)


def process_pois(poi_list: Iterable[dict], max_workers: int = 4, debug: bool = False) -> List[POIStructuredInfo]:
    pipeline = POIResearchPipeline()
    return pipeline.process_pois(poi_list, max_workers=max_workers, debug=debug)

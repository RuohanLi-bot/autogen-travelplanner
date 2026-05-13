"""
Travel Planner — mem0 + Neo4j edition.

Replaces FalkorDB with mem0 vector memory (user preferences) and a Neo4j
knowledge graph (Attraction → City → State, Attraction → Category, NEAR).

Usage:
    1. Run preload_memories.py once to ingest user review preferences.
    2. ``python main_mem0.py`` loads USER_QUERY from Open-AutoGLM ``task.py``, reuses the
       per-query cached XHS JSON if present, otherwise runs ``run_autoglm_bash_script()`` and
       caches the newly produced JSON before ingesting it.
    3. Xiaohongshu → graph: pass ``json_path=...`` or ``run_open_autoglm_first=True`` on
       ``add_autoglm_result_to_mem0`` as needed.

Environment (optional):
    MEM0_SKIP_PRELOAD=1 — 跳过把 review 写入 mem0（需已预载过或仅用无偏好流程）。
    若卡在 preload：mem0.add 会并行写 Qdrant + Neo4j，Neo4j 不可达时会长时间阻塞。
"""

import importlib.metadata as stdlib_importlib_metadata
import hashlib
import json
import logging
import os
import re
import select
import shlex
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from dotenv import load_dotenv
from pydantic import BaseModel, Field

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
PROJECT_ROOT = Path(__file__).resolve().parent
ENV_FILE = PROJECT_ROOT / ".env"
load_dotenv(ENV_FILE)   #override=False（默认），即不覆盖已有的环境变量

if not hasattr(stdlib_importlib_metadata, "packages_distributions"):
    try:
        import importlib_metadata as backport_importlib_metadata

        stdlib_importlib_metadata.packages_distributions = (
            backport_importlib_metadata.packages_distributions
        )
    except Exception:
        pass

from autogen import ConversableAgent, UserProxyAgent, LLMConfig
from autogen.agentchat import initiate_group_chat
from autogen.agentchat.group import (
    ReplyResult,
    ContextVariables,
    AgentTarget,
    AgentNameTarget,
    RevertToUserTarget,
    StayTarget,
    TerminateTarget,
    OnCondition,
    StringLLMCondition,
    StringAvailableCondition,
)
from autogen.agentchat.group.patterns import DefaultPattern
from autogen.agentchat.conversable_agent import UpdateSystemMessage

from mem0 import Memory

from itinerary_models import Day, Event, Itinerary, update_itinerary_with_travel_times
from graph_builder import (
    build_available_states,
    build_city_to_state_index,
    prepare_state_graph_data,
)
from preload_memories import preload_reviews
from poi_research.llm_client import OpenAILLMClient
from poi_research import POIStructuredInfo, process_pois
from critic_rules import evaluate_itinerary, validate_timed_itinerary

from stdout_log import stdout_to_log_file
from runtime_stats import runtime_stats
from xhs_travel_graph.graph_repository import Mem0Neo4jQueryRunner
from xhs_travel_graph.matcher import query_matching_play_modes
from xhs_travel_graph.models import MatchResult, TravelerProfile
from xhs_travel_graph.normalizer import stable_id
from xhs_travel_graph.pipeline import ingest_autoglm_json_to_structured_xhs_graph
from xhs_travel_graph.profile_parser import (
    parse_traveler_profile,
    profile_activity_key,
    profile_budget_level,
    profile_has_preference,
    profile_has_role,
    profile_is_mobility_limited,
    profile_max_age,
    profile_min_age,
)
from xhs_constraints.capability_graph_writer import CapabilityGraphWriter
from xhs_constraints.constraint_calibrator import (
    build_planning_budget,
    planning_budget_to_constraints,
    summarize_planning_budget,
)
from xhs_constraints.final_writer import write_final_itinerary
from xhs_constraints.models import CapabilityEstimate, MetricLimit
from xhs_constraints.optimizer import optimize_itinerary_from_play_modes
from xhs_constraints.playmode_fits import build_play_mode_fits, format_play_mode_fit_details, summarize_play_mode_fits
from xhs_constraints.query_semantics import (
    apply_constraint_specs_to_profile,
    generate_figure_mapping_questions,
    ground_constraint_spec_from_raw_result,
)
from xhs_constraints.scorer import format_scored_play_mode_details, score_play_modes, summarize_scored_play_modes
from xhs_constraints.validator import validate_final_itinerary

logger = logging.getLogger(__name__)

# Set in main() before chat / graph tools run
mem0_client = None
xhs_mem0_client = None
_available_states: Dict[str, str] = {}
_city_to_state: Dict[str, str] = {}
_city_lower_to_state: Dict[str, str] = {}

MAX_RETRY_ATTEMPTS = 5
MAX_CRITIC_ITERATIONS = MAX_RETRY_ATTEMPTS
MAX_RESTART_ATTEMPTS = MAX_RETRY_ATTEMPTS


def _quiet_third_party_loggers() -> None:
    """Suppress noisy INFO logs from third-party libraries while keeping warnings/errors."""
    noisy_loggers = [
        "httpx",
        "httpcore",
        "openai",
        "mem0",
        "mem0.vector_stores.qdrant",
        "qdrant_client",
    ]
    for name in noisy_loggers:
        logging.getLogger(name).setLevel(logging.WARNING)


def _install_openai_usage_patch() -> None:
    """Patch OpenAI chat.completions.create so all LLM calls share one usage counter."""
    try:
        from openai.resources.chat.completions import Completions
    except Exception as exc:
        logger.warning("OpenAI usage patch unavailable: %s", exc)
        return

    original_create = getattr(Completions.create, "_travel_planner_original", None)
    if original_create is not None:
        return

    original_create = Completions.create

    def counted_create(self, *args, **kwargs):
        try:
            response = original_create(self, *args, **kwargs)
        except Exception:
            runtime_stats.record_llm_call(success=False)
            raise
        usage = getattr(response, "usage", None)
        runtime_stats.record_llm_call(
            success=True,
            prompt_tokens=getattr(usage, "prompt_tokens", 0) if usage else 0,
            completion_tokens=getattr(usage, "completion_tokens", 0) if usage else 0,
            total_tokens=getattr(usage, "total_tokens", 0) if usage else 0,
        )
        return response

    counted_create._travel_planner_original = original_create  # type: ignore[attr-defined]
    Completions.create = counted_create

# =====================================================================
# 1. Configuration
# =====================================================================

api_key = os.environ.get("OPENAI_API_KEY")
openai_base_url = os.environ.get("OPENAI_BASE_URL")
neo4j_url = os.environ.get("NEO4J_URL")
neo4j_username = os.environ.get("NEO4J_USERNAME")
neo4j_password = os.environ.get("NEO4J_PASSWORD")
neo4j_database = os.environ.get("NEO4J_DATABASE")

llm_config = LLMConfig(
    model="gpt-4o-mini",
    api_key=api_key,
    base_url=openai_base_url,
    timeout=120,
    max_tokens=5000,
)
agent_llm_config = llm_config

USER_ID = os.environ.get("MEM0_USER_ID", "bryce_caster")

META_DIR = Path("/data/lrh/InteRecAgent/resources/google/googlelocal_data/business")

# =====================================================================
# 2. mem0 config (client created in main())
# =====================================================================

mem0_config = {
    "llm": {
        "provider": "openai",
        "config": {
            "model": "gpt-4o-mini",
            "api_key": api_key,
            "openai_base_url": openai_base_url,
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
            "url": neo4j_url,
            "username": neo4j_username,
            "password": neo4j_password,
            "database": neo4j_database,
        },
    },
}

REVIEWS_PATH = Path(
    "/data/lrh/InteRecAgent/resources/google/output/User/"
    "1_user_110127197526819446448/reviews.jsonl"
)
AUTOGLM_RESULT_PATH = Path(
    "/data/lrh/Open-AutoGLM-main/output/autoglm_run_20260408_101442.json"
)
# Open-AutoGLM repository root (fixed path).
AUTOGLM_ROOT = Path("/data/lrh/Open-AutoGLM-main")
XHS_QUERY_CACHE_DIR = AUTOGLM_ROOT / "output" / "query_cache"
FORCE_REFRESH_XHS_QUERY_CACHE = False
SKIP_INGEST_ON_CACHE_HIT = True
USE_EXISTING_GROUNDING_FILES = True
if str(AUTOGLM_ROOT) not in sys.path:
    sys.path.insert(0, str(AUTOGLM_ROOT))
from task import TASK, USER_QUERY


AUTOGLM_BROAD_OUTPUT_SCHEMA = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "task": {"type": "string"},
            "result": {"type": "string"},
        },
        "required": ["task", "result"],
    },
}

AUTOGLM_GROUNDING_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "question": {"type": "string"},
        "answer": {"type": "string"},
        "summary": {"type": "string"},
        "evidence": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "search_term": {"type": "string"},
                    "route_module": {"type": "string"},
                    "metric": {"type": "string"},
                    "value": {"type": ["number", "null"]},
                    "unit": {"type": "string"},
                    "direction": {"type": "string"},
                    "snippet": {"type": "string"},
                },
                "required": [
                    "search_term",
                    "route_module",
                    "metric",
                    "value",
                    "unit",
                    "direction",
                    "snippet",
                ],
            },
        },
    },
    "required": ["question", "answer", "summary", "evidence"],
}

AUTOGLM_GROUNDING_TIMEOUT_SEC = 900
AUTOGLM_GROUNDING_IDLE_TIMEOUT_SEC = 300
AUTOGLM_GROUNDING_MAX_SEARCH_PHRASES = 5
AUTOGLM_GROUNDING_MAX_POSTS = 10


def create_autoglm_run_dir(*, prefix: str = "run") -> Path:
    output_dir = AUTOGLM_ROOT / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_dir / f"{prefix}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


KNOWN_XHS_DESTINATIONS = (
    "张家界",
    "重庆",
    "北京",
    "三亚",
    "大理",
    "海南",
    "川西",
    "上海",
    "长沙",
    "成都",
    "西安",
    "杭州",
    "苏州",
    "南京",
    "厦门",
    "青岛",
    "广州",
    "深圳",
)

def _reset_planning_context(context_variables: ContextVariables) -> None:
    context_variables["destination_features"] = ""
    context_variables["user_preferences"] = ""
    context_variables["user_preferences_for_poi"] = ""
    context_variables["preference_retrieval_query"] = ""
    context_variables["preference_retrieval_needed"] = False
    context_variables["poi_candidates"] = []
    context_variables["poi_research_results"] = []
    context_variables["poi_research_markdown"] = ""
    context_variables["draft_itinerary"] = None
    context_variables["draft_itinerary_text"] = ""
    context_variables["critic_feedback"] = None
    context_variables["critic_iteration_count"] = 0
    context_variables["critic_stall_count"] = 0
    context_variables["critic_force_exit"] = ""
    context_variables["last_critic_draft"] = ""
    context_variables["last_critic_feedback_signature"] = ""
    context_variables["timed_itinerary_validation"] = None
    context_variables["planner_degraded"] = False
    context_variables["restart_attempt_count"] = 0
    context_variables.pop("timed_itinerary", None)
    context_variables["final_itinerary_presented_to_user"] = False
    context_variables["timed_itinerary_json_for_planner"] = ""
    # Block POI until planner finishes pref retrieval + set_poi_preference_summary.
    context_variables["poi_handoff_needed"] = False


def _build_meta_indices() -> None:
    logger.info("Building available-states index from %s ...", META_DIR)
    _available_states.clear()
    _available_states.update(build_available_states(META_DIR))
    logger.info("Index ready: %d states", len(_available_states))

    logger.info("Building city → state index from %s ...", META_DIR)
    _city_to_state.clear()
    _city_to_state.update(build_city_to_state_index(META_DIR))
    _city_lower_to_state.clear()
    _city_lower_to_state.update({k.lower(): v for k, v in _city_to_state.items()})
    logger.info("City→State index: %d cities", len(_city_to_state))


def _require_neo4j_config() -> None:
    missing = [
        name
        for name, value in (
            ("NEO4J_URL", neo4j_url),
            ("NEO4J_USERNAME", neo4j_username),
            ("NEO4J_PASSWORD", neo4j_password),
            ("NEO4J_DATABASE", neo4j_database),
        )
        if not value
    ]
    if missing:
        raise RuntimeError(
            "Missing required Neo4j configuration in .env/environment: " + ", ".join(missing)
        )


def _require_openai_config() -> None:
    if not api_key:
        raise RuntimeError("Missing required OPENAI_API_KEY in .env/environment.")


def _init_mem0_client() -> None:
    global mem0_client
    mem0_client = Memory.from_config(mem0_config)


def _get_xhs_mem0_client() -> Memory:
    cfg = json.loads(json.dumps(mem0_config))
    client = Memory.from_config(cfg)
    client.enable_graph = True
    return client


def _load_autoglm_result_texts(json_path: Path) -> List[str]:
    """Load AutoGLM JSON: one string per ``result`` entry (no concatenation).

    Supports:
    - A single object ``{"task": ..., "result": ...}`` → one-element list
    - An array of such objects (``save_task_result_json`` append format) → one element per item
    """
    with json_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    if isinstance(payload, list):
        out: List[str] = []
        for item in payload:
            if not isinstance(item, dict):
                continue
            r = item.get("result")
            if isinstance(r, str) and r.strip():
                out.append(r.strip())
        if not out:
            raise ValueError(f"No non-empty 'result' strings in array in {json_path}")
        return out

    if isinstance(payload, dict):
        result = payload.get("result")
        if isinstance(result, str) and result.strip():
            return [result.strip()]
        raise ValueError(f"'result' is missing or empty in {json_path}")
    raise ValueError(f"Unexpected JSON shape in {json_path}; expected object or array")


def run_autoglm_bash_script(
    *,
    extra_args: Optional[List[str]] = None,
    timeout_sec: Optional[int] = None,
    idle_timeout_sec: Optional[int] = None,
) -> int:
    """Run ``bash run_autoglm.sh`` under :data:`AUTOGLM_ROOT` (sets up PATH, Python, env).

    Pass ``extra_args`` to forward CLI flags to ``main.py`` (after the script's own handling).
    If ``extra_args`` is ``None``, ``MEM0_RUN_AUTOGLM_SH_ARGS`` or ``MEM0_AUTOGLM_ARGS`` is split
    with :func:`shlex.split` and appended (``MEM0_RUN_AUTOGLM_SH_ARGS`` takes precedence).
    """
    root = AUTOGLM_ROOT
    script = root / "run_autoglm.sh"
    if not script.is_file():
        raise FileNotFoundError(f"run_autoglm.sh not found: {script}")
    cmd: List[str] = ["bash", str(script)]
    if extra_args is not None:
        cmd.extend(extra_args)
    else:
        raw = (
            os.environ.get("MEM0_RUN_AUTOGLM_SH_ARGS", "").strip()
            or os.environ.get("MEM0_AUTOGLM_ARGS", "").strip()
        )
        if raw:
            cmd.extend(shlex.split(raw))
    logger.info("run_autoglm.sh: cwd=%s %s", root, " ".join(cmd))
    proc = subprocess.Popen(
        cmd,
        cwd=str(root),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None
    start_time = time.time()
    last_output_time = start_time
    try:
        while True:
            if proc.poll() is not None:
                break
            ready, _, _ = select.select([proc.stdout], [], [], 1.0)
            if ready:
                line = proc.stdout.readline()
                if line:
                    print(line, end="", file=sys.stderr, flush=True)
                    last_output_time = time.time()
            now = time.time()
            if timeout_sec is not None and now - start_time > timeout_sec:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                raise TimeoutError(f"AutoGLM exceeded total timeout {timeout_sec}s")
            if idle_timeout_sec is not None and now - last_output_time > idle_timeout_sec:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                raise TimeoutError(f"AutoGLM exceeded idle timeout {idle_timeout_sec}s")
        for line in proc.stdout:
            if line:
                print(line, end="", file=sys.stderr, flush=True)
        return proc.returncode
    finally:
        if proc.stdout is not None:
            proc.stdout.close()


def create_autoglm_output_json_path(*, prefix: str = "autoglm_run") -> Path:
    output_dir = AUTOGLM_ROOT / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return output_dir / f"{prefix}_{timestamp}.json"


def resolve_autoglm_grounding_cache_path(task_instruction: str) -> Path:
    output_dir = AUTOGLM_ROOT / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_key = hashlib.sha256(task_instruction.encode("utf-8")).hexdigest()[:16]
    return output_dir / f"autoglm_grounding_{cache_key}.json"


def load_cached_autoglm_grounding_result(
    *,
    task_instruction: str,
) -> Optional[Dict[str, Any]]:
    cache_path = resolve_autoglm_grounding_cache_path(task_instruction)
    if not cache_path.is_file():
        return None
    try:
        with cache_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception as exc:
        logger.warning("Failed to load cached grounding json %s: %s", cache_path, exc)
        return None
    if not isinstance(payload, dict):
        return None
    raw_result = payload.get("raw_result")
    if not isinstance(raw_result, str) or not raw_result.strip():
        return None
    print(f"[能力求证] 命中 grounding 缓存：{cache_path}", flush=True)
    return {
        "json_path": cache_path,
        "payload": payload,
    }


def load_existing_grounding_payloads() -> List[Dict[str, Any]]:
    output_dir = AUTOGLM_ROOT / "output"
    best_by_query: Dict[str, Dict[str, Any]] = {}
    for path in sorted(output_dir.glob("autoglm_grounding_*.json")):
        try:
            with path.open("r", encoding="utf-8") as f:
                payload = json.load(f)
        except Exception as exc:
            logger.warning("Failed to load grounding payload %s: %s", path, exc)
            continue
        if not isinstance(payload, dict):
            continue
        query = str(payload.get("query") or "").strip()
        raw_result = str(payload.get("raw_result") or "").strip()
        if not query or not raw_result:
            continue
        payload["_json_path"] = str(path)
        payload["_quality"] = _grounding_payload_quality(query, raw_result)
        current = best_by_query.get(query)
        if current is None or float(payload["_quality"]) >= float(current.get("_quality") or 0.0):
            best_by_query[query] = payload
    return list(best_by_query.values())


def _grounding_payload_quality(query: str, raw_result: str) -> float:
    text = raw_result or ""
    score = 0.0
    score += min(len(text), 2400) / 240.0
    score += 4.0 * len(re.findall(r"\d+(?:\.\d+)?\s*(?:公里|km|小时|h|分钟|min|步|级台阶|级阶)", text, re.I))
    score += 3.0 * len(re.findall(r"\d+\s*[-~到至]\s*\d+\s*(?:公里|km|小时|h|分钟|min|步|级台阶|级阶)", text, re.I))
    score += 1.5 * sum(1 for token in ("索道", "缆车", "电梯", "扶梯", "环保车", "景交车", "小火车") if token in text)
    score += 1.5 * sum(1 for token in ("排队", "拥挤", "错峰", "步行", "步数", "台阶", "爬升") if token in text)
    if any(token in text for token in ("AI助手", "问一问", "基于34篇笔记")):
        score -= 8.0
    if "相似画像游客" in query:
        score += 2.0 * len(re.findall(r"(?:步|公里|小时|分钟|台阶|阶梯)", text))
    return score


def find_cached_xhs_query_result_for_destination(destination: str) -> Optional[Dict[str, Any]]:
    destination = (destination or "").strip()
    if not destination:
        return None
    best_meta: Optional[Dict[str, Any]] = None
    best_meta_path: Optional[Path] = None
    for meta_path in sorted(XHS_QUERY_CACHE_DIR.glob("*.meta.json")):
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("Failed to load query cache meta %s: %s", meta_path, exc)
            continue
        if not isinstance(meta, dict):
            continue
        if str(meta.get("destination") or "").strip() != destination:
            continue
        if best_meta is None or str(meta.get("created_at") or "") >= str(best_meta.get("created_at") or ""):
            best_meta = meta
            best_meta_path = meta_path
    if best_meta is None or best_meta_path is None:
        return None
    cached_json = Path(str(best_meta.get("cached_json") or "")).expanduser()
    if not cached_json.is_file():
        return None
    return {
        "json_path": cached_json,
        "cache_hit": True,
        "cache_key": str(best_meta.get("cache_key") or ""),
        "cache_reused_by_destination": True,
        "meta_path": str(best_meta_path),
    }


def _estimate_from_traveler_profile(
    *,
    destination: str,
    traveler_profile_id: str,
    traveler_profile: TravelerProfile,
) -> CapabilityEstimate:
    metric_limits: List[MetricLimit] = []
    metric_map = {
        "walk_distance_km": ("walk_distance", "km"),
        "continuous_walk_min": ("continuous_walk_time", "min"),
        "stairs_steps": ("stairs_steps", "steps"),
        "active_duration_h": ("active_duration", "hours"),
        "queue_time_min": ("queue_exposure", "min"),
        "elevation_gain_m": ("elevation_gain_m", "m"),
    }
    for item in traveler_profile.strength or []:
        if not isinstance(item, dict):
            continue
        metric_key = str(item.get("metric_key") or "").strip()
        metric_info = metric_map.get(metric_key)
        if not metric_info:
            continue
        value = item.get("value")
        if not isinstance(value, (int, float)):
            continue
        op = str(item.get("op") or "").strip()
        if op not in {"<=", "=="}:
            continue
        is_hard = bool(item.get("hard", False))
        metric, default_unit = metric_info
        soft_limit = None
        hard_limit = None
        if op == "==":
            soft_limit = float(value)
            hard_limit = float(value) if is_hard else None
        else:
            if is_hard:
                hard_limit = float(value)
            else:
                soft_limit = float(value)
        metric_limits.append(
            MetricLimit(
                metric=metric,
                scenario="default",
                soft_limit=soft_limit,
                hard_limit=hard_limit,
                unit=default_unit,
                confidence=0.0,
                evidence_count=0,
                notes=[str(item.get("description") or "traveler_profile")],
            )
        )
    preferred_tags: Dict[str, float] = {}
    if profile_has_role(traveler_profile, "senior") or profile_is_mobility_limited(traveler_profile):
        preferred_tags["accessibility"] = 0.75
    if profile_has_role(traveler_profile, "child"):
        preferred_tags["family_friendly"] = 0.55
    return CapabilityEstimate(
        estimate_id=stable_id("traveler_profile_estimate", traveler_profile_id, destination),
        destination=destination,
        traveler_profile_id=traveler_profile_id,
        metric_limits=metric_limits,
        forbidden_same_day_pairs=[],
        discouraged_same_day_pairs=[],
        preferred_tags=preferred_tags,
        notes=[f"profile_source={traveler_profile.source}"],
        confidence=0.0,
    )


def run_autoglm_cli_task(
    *,
    task_instruction: str,
    output_json_path: Path,
    search_query: str = "",
    output_schema: Optional[Dict[str, Any]] = None,
    timeout_sec: Optional[int] = None,
    idle_timeout_sec: Optional[int] = None,
) -> Dict[str, Any]:
    """Call Open-AutoGLM via CLI with explicit structured parameters."""
    output_json_path = Path(output_json_path).expanduser().resolve()
    cmd_args: List[str] = [
        "--task-instruction",
        task_instruction,
        "--output-json",
        str(output_json_path),
    ]
    if search_query.strip():
        cmd_args.extend(["--search-query", search_query.strip()])
    if output_schema:
        cmd_args.extend(
            ["--output-schema", json.dumps(output_schema, ensure_ascii=False)]
        )
    code = run_autoglm_bash_script(
        extra_args=cmd_args,
        timeout_sec=timeout_sec,
        idle_timeout_sec=idle_timeout_sec,
    )
    if code != 0:
        raise SystemExit(code)
    if not output_json_path.is_file():
        raise FileNotFoundError(f"AutoGLM output JSON not found: {output_json_path}")
    with output_json_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    return {
        "json_path": output_json_path,
        "payload": payload,
    }


def build_grounding_task_instruction(
    *,
    figure: List[str],
    description: str,
) -> str:
    figure_text = "、".join(str(item).strip() for item in figure if str(item).strip()) or "当前人群"
    base = (
        "你需要围绕下面这条用户画像约束说明，在小红书中主动搜索关键词、查看相关图文笔记，"
        "寻找可以量化或确认这条约束/偏好的证据。\n"
        f"【figure】\n{figure_text}\n\n"
        f"【待求证说明】\n{description}\n\n"
        "执行要求：\n"
        "1. 根据 figure 和待求证说明，自行设计若干搜索表达，不要机械重复单一关键词。\n"
        "2. 重点寻找能够支持数值、范围、偏好集合或明确结论的帖子内容。\n"
        "3. 如果证据不足，请保留 uncertain，不要编造。\n"
        "4. 最终必须只输出 JSON，不要输出额外解释。\n"
        "最终 JSON 字段要求：\n"
        'question: 当前待求证说明；answer: supported|unsupported|mixed|uncertain；summary: 简短总结；evidence: 1-5 条关键证据。'
    )
    return (
        f"{base}\n"
        "额外执行预算：\n"
        f"1. 最多尝试 {AUTOGLM_GROUNDING_MAX_SEARCH_PHRASES} 组搜索表达。\n"
        f"2. 总浏览帖子数不超过 {AUTOGLM_GROUNDING_MAX_POSTS} 篇。\n"
        "3. 一旦已经有足够证据能够回答问题，就立即停止继续搜索。\n"
        "4. 如果证据仍不足，请输出 uncertain，不要无限继续尝试。"
    )


def _resolve_autoglm_output_json_path(output_dir: Path, since: float) -> Path:
    """Pick the JSON file produced by the latest ``main.py`` run (by mtime)."""
    if not output_dir.is_dir():
        raise FileNotFoundError(f"Open-AutoGLM output directory not found: {output_dir}")
    files = list(output_dir.glob("autoglm_run_*.json"))
    if not files:
        raise FileNotFoundError(f"No autoglm_run_*.json under {output_dir}")
    fresh = [p for p in files if p.stat().st_mtime >= since - 1.0]
    pool = fresh if fresh else files
    return max(pool, key=lambda p: p.stat().st_mtime)

def _normalize_query_for_cache(user_query: str) -> str:
    return re.sub(r"\s+", " ", (user_query or "").strip())


def xhs_query_cache_key(user_query: str) -> str:
    normalized = _normalize_query_for_cache(user_query)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]


def resolve_xhs_query_cache_path(user_query: str) -> Path:
    return XHS_QUERY_CACHE_DIR / f"xhs_{xhs_query_cache_key(user_query)}.json"


def resolve_xhs_query_cache_meta_path(user_query: str) -> Path:
    return XHS_QUERY_CACHE_DIR / f"xhs_{xhs_query_cache_key(user_query)}.meta.json"


def save_xhs_query_cache(
    *,
    source_json: Path,
    user_query: str,
    destination: str,
    trip_days: int,
) -> Path:
    source_json = Path(source_json).expanduser().resolve()
    if not source_json.is_file():
        raise FileNotFoundError(f"AutoGLM output JSON not found: {source_json}")

    XHS_QUERY_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    target_json = resolve_xhs_query_cache_path(user_query)
    target_meta = resolve_xhs_query_cache_meta_path(user_query)
    shutil.copy2(source_json, target_json)

    cache_meta = {
        "cache_key": xhs_query_cache_key(user_query),
        "user_query": _normalize_query_for_cache(user_query),
        "destination": destination,
        "trip_days": trip_days,
        "source_json": str(source_json),
        "cached_json": str(target_json),
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    target_meta.write_text(
        json.dumps(cache_meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return target_json


def get_or_create_xhs_query_result_json(
    *,
    user_query: str,
    destination: str,
    trip_days: int,
    force_refresh: bool = FORCE_REFRESH_XHS_QUERY_CACHE,
) -> Dict[str, Any]:
    cache_path = resolve_xhs_query_cache_path(user_query)
    if cache_path.is_file() and not force_refresh:
        print(f"[小红书查询] 命中缓存，直接使用结果文件：{cache_path}", flush=True)
        return {
            "json_path": cache_path,
            "cache_hit": True,
            "cache_key": xhs_query_cache_key(user_query),
        }

    output_dir = AUTOGLM_ROOT / "output"
    print("[小红书查询] 未命中缓存，开始调用 Open-AutoGLM 在线检索小红书。", flush=True)
    latest_json = create_autoglm_output_json_path(prefix="autoglm_run")
    run_autoglm_cli_task(
        task_instruction=TASK,
        search_query=user_query,
        output_schema=AUTOGLM_BROAD_OUTPUT_SCHEMA,
        output_json_path=latest_json,
    )
    if not latest_json.is_file():
        latest_json = _resolve_autoglm_output_json_path(output_dir, time.time())
    cached_json = save_xhs_query_cache(
        source_json=latest_json,
        user_query=user_query,
        destination=destination,
        trip_days=trip_days,
    )
    print(f"[小红书查询] 新结果已写入缓存：{cached_json}", flush=True)
    return {
        "json_path": cached_json,
        "cache_hit": False,
        "cache_key": xhs_query_cache_key(user_query),
        "source_json": latest_json,
    }


def add_autoglm_result_to_mem0(
    json_path: Optional[Path] = None,
    mem0_client: Optional[Memory] = None,
    user_id: str = USER_ID,
    metadata: Optional[Dict[str, Any]] = None,
    vector_infer: bool = False,
    write_structured_graph: bool = True,
    destination: str = "",
) -> Dict[str, Any]:
    """
    把帖子文本写进向量库、抽成结构化图写入 Neo4j：RouteVariant、Place、Constraint、Risk、RouteSegment
    把多个 RouteVariant 聚类成PlayMode玩法簇
    """
    json_path = Path(json_path).expanduser().resolve()
    print(f"[小红书入库] 读取 AutoGLM 结果文件：{json_path}", flush=True)
    result_texts = _load_autoglm_result_texts(json_path)
    ingest_run_id = "xhs"
    print(f"[小红书入库] 待处理帖子/结果条数：{len(result_texts)}", flush=True)

    # base_meta: Dict[str, Any] = {
    #     "source": "autoglm_result",
    #     "source_file": str(json_path),
    #     "ingest_field": "result",
    #     "graph_label": "xhs",
    #     "timestamp": int(time.time()),
    # }
    # if metadata:
    #     base_meta.update(metadata)

    # responses: List[Any] = []
    # n = len(result_texts)
    # print("[小红书入库] 开始写入向量和图数据库。", flush=True)
    # for idx, result_text in enumerate(result_texts):
    #     chunk_meta = {
    #         **base_meta,
    #         "result_index": idx,
    #         "result_count": n,
    #     }
    #     r = mem0_client.add(
    #         messages=[{"role": "user", "content": result_text}],
    #         user_id=user_id,
    #         run_id=ingest_run_id,
    #         metadata=chunk_meta,
    #         infer=vector_infer,
    #     )
    #     responses.append(r)
    # print(f"[向量数据库] 本次新增数据：{len(responses)} 条。", flush=True)

    structured_result: Dict[str, Any] = {}
    if write_structured_graph:
        print("[小红书入库] 开始写入结构化图谱并聚类玩法簇。", flush=True)
        structured_result = ingest_autoglm_json_to_structured_xhs_graph(
            json_path=json_path,
            mem0_client=mem0_client,
            run_id=ingest_run_id,
            destination=destination,
        )
        print(
            f"[小红书入库] 结构化图谱处理完成：{_structured_result_summary_for_print(structured_result)}",
            flush=True,
        )

def run_autoglm_grounding_question(
    *,
    figure: List[str],
    description: str,
) -> Optional[str]:
    """Run one grounding question through Open-AutoGLM without fixed search-query."""
    try:
        task_instruction = build_grounding_task_instruction(
            figure=figure,
            description=description,
        )
        result = load_cached_autoglm_grounding_result(
            task_instruction=task_instruction,
        )
        if result is None:
            output_json = resolve_autoglm_grounding_cache_path(task_instruction)
            result = run_autoglm_cli_task(
                task_instruction=task_instruction,
                output_schema=AUTOGLM_GROUNDING_OUTPUT_SCHEMA,
                output_json_path=output_json,
                timeout_sec=AUTOGLM_GROUNDING_TIMEOUT_SEC,
                idle_timeout_sec=AUTOGLM_GROUNDING_IDLE_TIMEOUT_SEC,
            )
        payload = result.get("payload")
        if isinstance(payload, dict):
            raw_result = payload.get("raw_result")
            if isinstance(raw_result, str) and raw_result.strip():
                return raw_result
    except TimeoutError as exc:
        print(f"[能力求证] AutoGLM grounding 超时，回退本地求证：{exc}", flush=True)
    except Exception as exc:
        print(f"[能力求证] AutoGLM grounding 失败，回退本地求证：{exc}", flush=True)
    return None


def parse_xhs_task_request(user_query: str) -> Dict[str, Any]:
    query = (user_query or "").strip()
    destination = _parse_xhs_destination(query)
    trip_days = _parse_xhs_trip_days(query)
    return {
        "user_query": query,
        "destination": destination,
        "trip_days": trip_days,
    }


def _parse_xhs_destination(user_query: str) -> str:
    for destination in KNOWN_XHS_DESTINATIONS:
        if destination in user_query:
            return destination
    patterns = [
        r"(?:前往|去|游览|到)([\u4e00-\u9fa5]{2,8})(?:，|,|。|旅|玩|游|$)",
        r"([\u4e00-\u9fa5]{2,8})(?:旅游攻略|攻略|两天|三天|四天|五天)",
    ]
    for pattern in patterns:
        match = re.search(pattern, user_query)
        if match:
            candidate = match.group(1).strip()
            candidate = re.sub(r"^(计划|想|希望|高效游览)", "", candidate).strip()
            if candidate:
                return candidate
    return ""


def _parse_xhs_trip_days(user_query: str) -> int:
    match = re.search(r"(\d+)\s*(?:天|日)", user_query)
    if match:
        return max(1, int(match.group(1)))
    chinese_day_match = re.search(r"([一二两三四五六七八九十]+)\s*(?:天|日)", user_query)
    if chinese_day_match:
        return max(1, _chinese_number_to_int(chinese_day_match.group(1)))
    if "周末" in user_query:
        return 2
    return 2


def _chinese_number_to_int(value: str) -> int:
    digits = {
        "一": 1,
        "二": 2,
        "两": 2,
        "三": 3,
        "四": 4,
        "五": 5,
        "六": 6,
        "七": 7,
        "八": 8,
        "九": 9,
    }
    if value == "十":
        return 10
    if "十" in value:
        left, _, right = value.partition("十")
        tens = digits.get(left, 1) if left else 1
        ones = digits.get(right, 0) if right else 0
        return tens * 10 + ones
    return digits.get(value, 3)


def _structured_result_summary_for_print(result: Dict[str, Any]) -> str:
    if not result:
        return "无结构化结果"
    parts = []
    for key in ("post_count", "fact_count", "route_variant_count", "play_mode_count"):
        if key in result:
            parts.append(f"{key}={result[key]}")
    cluster = result.get("cluster_summary")
    if isinstance(cluster, dict):
        for key in ("play_mode_count", "route_variant_count"):
            if key in cluster:
                parts.append(f"cluster_{key}={cluster[key]}")
    return "，".join(parts) or "已完成"


def _profile_summary_for_print(profile: Any) -> str:
    parts = []
    figures = getattr(profile, "figure", []) or []
    if figures:
        parts.append("figure=" + "、".join(str(item) for item in figures))
    activity = _profile_spec_values(getattr(profile, "activity", []) or [])
    if activity:
        parts.append("活动=" + "、".join(activity[:3]))
    preference = _profile_spec_values(getattr(profile, "preference", []) or [])
    if preference:
        parts.append("偏好=" + "、".join(preference[:3]))
    strength_keys = _profile_spec_metric_keys(getattr(profile, "strength", []) or [])
    if strength_keys:
        parts.append("strength=" + "、".join(strength_keys[:5]))
    budget = profile_budget_level(profile) if isinstance(profile, TravelerProfile) else "unknown"
    parts.append(f"预算={budget}")
    return "；".join(parts)

def _traveler_profile_mapping_summary(profile: TravelerProfile) -> str:
    strength_keys = _profile_spec_metric_keys(profile.strength or [])
    return (
        f"figure={','.join(profile.figure) or '无'}；"
        f"预算={profile_budget_level(profile)}；"
        f"活动={','.join(_profile_spec_values(profile.activity or [])) or '无'}；"
        f"偏好={','.join(_profile_spec_values(profile.preference or [])) or '无'}；"
        f"strength={','.join(strength_keys) or '无'}"
    )


def _profile_spec_metric_keys(items: List[Any]) -> List[str]:
    keys: List[str] = []
    seen = set()
    for item in items:
        if not isinstance(item, dict):
            continue
        key = str(item.get("metric_key") or "").strip()
        if not key or key in seen:
            continue
        seen.add(key)
        keys.append(key)
    return sorted(keys)


def _profile_spec_values(items: List[Any]) -> List[str]:
    values: List[str] = []
    seen = set()
    for item in items:
        if not isinstance(item, dict):
            continue
        raw = item.get("value")
        normalized_values = raw if isinstance(raw, list) else [raw]
        for value in normalized_values:
            text = str(value or "").strip()
            if not text or text in seen:
                continue
            seen.add(text)
            values.append(text)
    return values


def _match_summary_for_print(matches: List[MatchResult]) -> str:
    counts: Dict[str, int] = {}
    for match in matches:
        decision = getattr(match.assessment, "decision", "unknown")
        counts[decision] = counts.get(decision, 0) + 1
    ordered = ["pass", "conditional", "unknown", "fail"]
    body = "，".join(f"{key}={counts.get(key, 0)}" for key in ordered)
    return f"总数={len(matches)}，{body}"


def _format_limit_for_print(label: str, limit: Any) -> str:
    soft = getattr(limit, "soft", None)
    hard = getattr(limit, "hard", None)
    unit = getattr(limit, "unit", "")
    confidence = getattr(limit, "confidence", None)
    evidence_count = getattr(limit, "evidence_count", None)
    source = getattr(limit, "source", None)
    source_text = ""
    if isinstance(source, list) and source:
        source_text = f"，来源={','.join(str(item) for item in source)}"
    elif isinstance(source, str) and source:
        source_text = f"，来源={source}"
    confidence_text = f"，置信度={confidence:.2f}" if isinstance(confidence, float) else ""
    evidence_text = f"，证据数={evidence_count}" if isinstance(evidence_count, int) else ""
    return f"{label}: soft={soft}{unit}, hard={hard}{unit}{confidence_text}{evidence_text}{source_text}"


def _skeleton_summary_for_print(skeleton: Any) -> str:
    attraction_count = 0
    rest_count = 0
    for day in getattr(skeleton, "days", []) or []:
        for event in getattr(day, "events", []) or []:
            if getattr(event, "type", "") == "Attraction":
                attraction_count += 1
            elif getattr(event, "type", "") == "Rest":
                rest_count += 1
    return f"天数={len(getattr(skeleton, 'days', []) or [])}，主景点={attraction_count}，休息缓冲={rest_count}"


def run_xhs_full_itinerary_flow(
    *,
    user_query: str,
    destination: str,
    trip_days: int,
    autoglm_ingest_result: Optional[Dict[str, Any]] = None,
    mem0_client: Optional[Memory] = None,
) -> Dict[str, Any]:
    profile = parse_traveler_profile(user_query, destination=destination)
    print(f"[行程生成] 解析用户画像：{_profile_summary_for_print(profile)}", flush=True)
    runner = Mem0Neo4jQueryRunner(mem0_client)
    capability_writer = CapabilityGraphWriter(runner)
    generation_context: Dict[str, Any] = {"mode": "play_mode_capability_planning"}
    try:
        constraint_specs: List[Dict[str, Any]] = []
        grounded_specs: List[Dict[str, Any]] = []
        #从图中抽出详细TravelerProfile
        reused_profile = capability_writer.load_traveler_profile_by_figure(
            user_id=USER_ID,
            figure=profile.figure,
        )
        if reused_profile:
            profile = reused_profile.model_copy(
                update={
                    "destination": destination,
                    "user_query": user_query,
                    "source": "graph_reuse",
                }
            )
            print(f"[画像复用] 命中 figure 画像，已加载完整 TravelerProfile：{_traveler_profile_mapping_summary(profile)}", flush=True)
        else:
            print("[画像复用] 未命中 figure 画像，将进入 figure 细粒度偏好映射求证。", flush=True)
            constraint_specs = generate_figure_mapping_questions(
                traveler_profile=profile,
            )
            print(
                f"[约束生成] figure={','.join(profile.figure) or '无'}；spec数={len(constraint_specs)}",
                flush=True,
            )
            llm_client = OpenAILLMClient()
            for spec in constraint_specs:
                raw_result = run_autoglm_grounding_question(
                    figure=profile.figure,
                    description=str(spec.get("description") or ""),
                )
                grounded_specs.append(
                    ground_constraint_spec_from_raw_result(
                        spec=spec,
                        raw_result=raw_result or "",
                        llm_client=llm_client,
                    )
                )
            print(f"[约束求证] 完成 grounded specs：{len(grounded_specs)} 条。", flush=True)
            profile = apply_constraint_specs_to_profile(
                traveler_profile=profile,
                specs=grounded_specs,
                source="grounded",
            )
            capability_writer.write_traveler_profile(user_id=USER_ID, traveler_profile=profile)
            print(f"[画像补全] TravelerProfile 已补齐：{_traveler_profile_mapping_summary(profile)}", flush=True)
            print(f"[画像写图] 已写入 TravelerProfile：{_traveler_profile_mapping_summary(profile)}", flush=True)
        capability_estimate = _estimate_from_traveler_profile(
            destination=destination,
            traveler_profile_id=profile.profile_id,
            traveler_profile=profile,
        )
        print(f"[画像摘要] 规划前画像：{_traveler_profile_mapping_summary(profile)}", flush=True)
        print(f"[玩法簇匹配] 开始从本地图谱查询目的地「{destination}」的候选玩法簇。", flush=True)
        matches = query_matching_play_modes(
            query_runner=runner,
            run_id="xhs",
            destination=destination,
            profile=profile,
            write_assessments=True,
            include_blocked=False,
            limit=max(10, trip_days * 3),
        )
        print(f"[玩法簇匹配] 匹配完成：{_match_summary_for_print(matches)}", flush=True)
        planning_budget = build_planning_budget(
            estimate=capability_estimate,
            traveler_profile=profile,
        )
        print(f"[规划预算] {summarize_planning_budget(planning_budget)}", flush=True)
        active_constraints = planning_budget_to_constraints(
            planning_budget=planning_budget,
            traveler_profile=profile,
        )
        generation_context["traveler_profile"] = _model_to_dict(profile)

        play_mode_fits = build_play_mode_fits(
            query_runner=runner,
            matches=matches,
            planning_budget=planning_budget,
        )
        print(f"[玩法拟合] {summarize_play_mode_fits(play_mode_fits)}", flush=True)
        print(format_play_mode_fit_details(play_mode_fits), flush=True)

        scored_play_modes = score_play_modes(
            play_mode_fits=play_mode_fits,
            planning_budget=planning_budget,
        )
        print(f"[候选评分] {summarize_scored_play_modes(scored_play_modes)}", flush=True)
        print(format_scored_play_mode_details(scored_play_modes), flush=True)

        skeleton = optimize_itinerary_from_play_modes(
            scored_play_modes=scored_play_modes,
            planning_budget=planning_budget,
            constraints=active_constraints,
            destination=destination,
            trip_days=trip_days,
        )
        print(f"[行程优化] 行程骨架生成完成：{_skeleton_summary_for_print(skeleton)}", flush=True)
        itinerary = write_final_itinerary(skeleton=skeleton)
        print("[最终输出] 已根据行程骨架生成最终行程文本。", flush=True)
        validation_report = validate_final_itinerary(
            itinerary=itinerary,
            skeleton=skeleton,
            constraints=active_constraints,
        )
        print(f"[行程校验] 发现问题数量：{len(validation_report.issues)}", flush=True)
        generation_context.update(
            {
                "traveler_profile": _model_to_dict(profile),
                "constraint_specs": constraint_specs,
                "grounded_specs": grounded_specs,
                "capability_estimate": _model_to_dict(capability_estimate),
                "planning_budget": _model_to_dict(planning_budget),
                "active_constraints": _model_to_dict(active_constraints),
                "play_mode_fits": [_model_to_dict(item) for item in play_mode_fits],
                "scored_play_modes": [_model_to_dict(item) for item in scored_play_modes],
                "itinerary_skeleton": _model_to_dict(skeleton),
                "validation_report": _model_to_dict(validation_report),
            }
        )
        if validation_report.issues:
            print(
                "[行程校验] 问题详情："
                + "; ".join(issue.message for issue in validation_report.issues),
                flush=True,
            )
    except Exception as exc:
        logger.exception("Constraint-aware XHS itinerary generation failed; falling back.")
        print(
            f"[行程生成] 约束式生成失败：{exc}。降级为直接展开玩法簇。",
            flush=True,
        )
        generation_context = {"mode": "degraded_play_mode_fallback", "error": str(exc)}
        itinerary = build_xhs_itinerary_from_matches(
            matches=matches,
            destination=destination,
            trip_days=trip_days,
            user_query=user_query,
        )
    result = {
        "user_query": user_query,
        "destination": destination,
        "trip_days": trip_days,
        "traveler_profile": _model_to_dict(profile),
        "matches": [_model_to_dict(match) for match in matches],
        "itinerary": _model_to_dict(itinerary),
        "generation_context": generation_context,
        "autoglm_ingest_result": autoglm_ingest_result or {},
    }
    print(format_xhs_match_summary(matches, destination, trip_days), flush=True)
    print_itinerary(result["itinerary"])
    return result


def build_xhs_itinerary_from_matches(
    *,
    matches: List[MatchResult],
    destination: str,
    trip_days: int,
    user_query: str,
) -> Itinerary:
    usable_matches = [match for match in matches if not match.blocked_by_safety_floor and match.assessment.decision != "fail"]
    if not usable_matches:
        return Itinerary(
            days=[
                Day(
                    events=[
                        Event(
                            type="Attraction",
                            location=destination,
                            city=destination,
                            description=(
                                "未从小红书结构化图谱中找到可直接推荐的玩法簇。"
                                f"用户需求：{user_query}。建议人工核实后再执行。"
                            ),
                        )
                    ]
                )
                for _ in range(trip_days)
            ]
        )

    days: List[Day] = []
    used_locations = set()
    for day_index in range(trip_days):
        match = usable_matches[day_index % len(usable_matches)]
        raw = match.raw or {}
        places = _pick_day_places(raw, fallback=match.name, used_locations=used_locations)
        events: List[Event] = []
        for place_index, place in enumerate(places):
            if place_index > 0:
                events.append(
                    Event(
                        type="Travel",
                        location=f"景区内移动至 {place}",
                        city=destination,
                        description=_xhs_travel_description(raw),
                    )
                )
            events.append(
                Event(
                    type="Attraction",
                    location=place,
                    city=destination,
                    description=_xhs_event_description(
                        match=match,
                        day_index=day_index + 1,
                        place=place,
                    ),
                    poi_id=match.play_mode_id,
                    itinerary_role="xhs_play_mode",
                )
            )
        days.append(Day(events=events))
    return Itinerary(days=days)


def _pick_day_places(raw: Dict[str, Any], *, fallback: str, used_locations: set) -> List[str]:
    places = [str(place).strip() for place in raw.get("representative_places") or [] if str(place).strip()]
    if not places:
        places = _places_from_play_mode_name(fallback)
    if not places:
        places = [fallback or "待核实玩法"]
    selected = []
    for place in places:
        if place in used_locations and len(selected) >= 1:
            continue
        selected.append(place)
        used_locations.add(place)
        if len(selected) >= 3:
            break
    return selected or places[:2]


def _places_from_play_mode_name(name: str) -> List[str]:
    parts = [part for part in re.split(r"[-/]", name or "") if part]
    filtered = [
        part
        for part in parts
        if part not in {"目的地", "常规交通", "玩法", "亲子轻松线", "高强度打卡线", "省钱线"}
        and len(part) <= 16
    ]
    return filtered[:3]


def _xhs_travel_description(raw: Dict[str, Any]) -> str:
    modes = [str(mode) for mode in raw.get("dominant_transport_modes") or [] if mode]
    if modes:
        return "根据小红书玩法簇，优先使用景区内交通方式：" + "、".join(modes[:3]) + "。具体耗时以现场为准。"
    return "根据小红书玩法簇安排景区内移动；具体交通和耗时需现场核实。"


def _xhs_event_description(*, match: MatchResult, day_index: int, place: str) -> str:
    assessment = match.assessment
    parts = [
        f"第{day_index}天采用玩法簇：{match.name}。",
        f"适配结论：{assessment.decision}。",
    ]
    if assessment.reasons:
        parts.append("原因：" + "；".join(assessment.reasons[:2]) + "。")
    if assessment.required_actions:
        parts.append("必须执行：" + "；".join(assessment.required_actions[:3]) + "。")
    if assessment.missing_evidence:
        parts.append("证据缺口：" + "；".join(assessment.missing_evidence[:3]) + "，需现场核实。")
    if assessment.evidence_used:
        evidence = "；".join(_shorten_text(item, 80) for item in assessment.evidence_used[:2])
        parts.append("小红书证据：" + evidence)
    return "".join(parts)


def format_xhs_match_summary(matches: List[MatchResult], destination: str, trip_days: int) -> str:
    lines = [
        "",
        "=" * 80,
        f"XHS PlayMode matches for {destination} ({trip_days} days)",
        "=" * 80,
    ]
    if not matches:
        lines.append("No PlayMode match found. The generated itinerary will be a degraded fallback.")
        return "\n".join(lines)
    for idx, match in enumerate(matches[:8], 1):
        lines.extend(
            [
                f"{idx}. {match.name}",
                f"   decision={match.assessment.decision}, hard_fail={match.assessment.hard_fail}, evidence_count={match.evidence_count}",
                f"   reasons={'；'.join(match.assessment.reasons[:2]) or 'n/a'}",
                f"   required_actions={'；'.join(match.assessment.required_actions[:2]) or 'n/a'}",
                f"   missing_evidence={'；'.join(match.assessment.missing_evidence[:2]) or 'n/a'}",
            ]
        )
    lines.append("=" * 80)
    return "\n".join(lines)


def _shorten_text(text: str, limit: int) -> str:
    raw = re.sub(r"\s+", " ", str(text or "")).strip()
    return raw if len(raw) <= limit else raw[:limit] + "..."


def _model_to_dict(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if isinstance(value, list):
        return [_model_to_dict(item) for item in value]
    if isinstance(value, dict):
        return {key: _model_to_dict(item) for key, item in value.items()}
    return value


def _prepare_mem0_runtime(client: Optional[Memory] = None) -> None:
    # 群聊启动前准备 mem0：清空本地 Qdrant 集合，再把用户评论预载进向量库。
    runtime_client = client or mem0_client
    if runtime_client is None:
        raise RuntimeError("_prepare_mem0_runtime() requires an initialized mem0 client.")
    logger.info("Resetting local Qdrant collection (mem0 vector_store.reset) ...")
    vector_store = runtime_client.vector_store
    reset = getattr(vector_store, "reset", None)
    if callable(reset):
        reset()
    else:
        logger.warning("Vector store has no reset(); skipping Qdrant clear.")
    logger.info("Preloading user review preferences into mem0 ...")
    print(
        "若长时间停在此处，多半是 mem0.add 在等待 Neo4j（bolt）或本地 embedding；"
        "可检查 NEO4J_URL 网络与防火墙。跳过预加载可设环境变量 MEM0_SKIP_PRELOAD=1。",
        file=sys.stderr,
        flush=True,
    )
    preload_reviews(runtime_client, reviews_path=REVIEWS_PATH, user_id=USER_ID)


def _resolve_state_from_city(city: str) -> Optional[str]:
    """Resolve US state from a city name using meta-data index (case-insensitive)."""
    c = (city or "").strip()
    if not c:
        return None
    if c in _city_to_state:
        return _city_to_state[c]
    return _city_lower_to_state.get(c.lower())


# =====================================================================
# 4. Context variables & tool functions
# =====================================================================

# user_preferences: raw semantic hits from mem0.search (pref_agent).
# user_preferences_for_poi: short summary written by planner_agent for poi embedding/scoring.
trip_context = ContextVariables(
    data={
        "itinerary_confirmed": False,
        "structured_itinerary": None,
        "draft_itinerary_text": "",
        "user_preferences": "",
        "user_preferences_for_poi": "",
        "preference_retrieval_query": "",
        "preference_retrieval_needed": False,
        "destination_state": None,
        "destination_city": None,
        "destination_features": "",
        # True after set_poi_preference_summary until poi_agent returns candidates.
        "poi_handoff_needed": False,
        "pending_preference": "",
        "preference_storage_needed": False,
        "poi_candidates": [],
        "poi_research_results": [],
        "poi_research_markdown": "",
        "draft_itinerary": None,
        "critic_feedback": None,
        "critic_iteration_count": 0,
        "critic_stall_count": 0,
        "critic_force_exit": "",
        "last_critic_draft": "",
        "last_critic_feedback_signature": "",
        "timed_itinerary_validation": None,
        "planner_degraded": False,
        "restart_attempt_count": 0,
        "last_user_feedback": "",
        "last_user_feedback_empty": False,
        "final_itinerary_presented_to_user": False,
        "timed_itinerary_json_for_planner": "",
    }
)


# ---- Preference tools ----

def queue_user_preference_storage(
    preference: str, context_variables: ContextVariables
) -> ReplyResult:
    """Queue a new preference so pref_agent persists it in mem0 with metadata."""
    context_variables["pending_preference"] = preference.strip()
    context_variables["preference_storage_needed"] = True
    return ReplyResult(
        message=f"Queued preference for storage: {preference}",
        context_variables=context_variables,
        target=StayTarget(),
    )


PREF_SEMANTIC_SEARCH_LIMIT = 15


def queue_preference_retrieval(
    query: str, context_variables: ContextVariables
) -> ReplyResult:
    """Queue semantic retrieval: pref_agent will call mem0.search(query, user_id=...)."""
    q = (query or "").strip()
    if not q:
        return ReplyResult(
            message="preference retrieval query must be non-empty.",
            context_variables=context_variables,
            target=StayTarget(),
        )
    context_variables["preference_retrieval_query"] = q
    context_variables["preference_retrieval_needed"] = True
    context_variables["user_preferences_for_poi"] = ""
    context_variables["poi_handoff_needed"] = False
    return ReplyResult(
        message="Queued semantic preference retrieval for pref_agent (mem0.search).",
        context_variables=context_variables,
        target=StayTarget(),
    )


def set_poi_preference_summary(
    summary: str, context_variables: ContextVariables
) -> ReplyResult:
    """Store the planner's short summary of retrieved prefs for poi_agent embedding/scoring."""
    context_variables["user_preferences_for_poi"] = (summary or "").strip()
    context_variables["poi_handoff_needed"] = True
    return ReplyResult(
        message=(
            "POI preference summary stored. Hand off to poi_agent once to load candidates "
            "(only while poi_handoff_needed is true)."
        ),
        context_variables=context_variables,
        target=StayTarget(),
    )


def set_destination_features(
    destination_features: str,
    context_variables: ContextVariables,
) -> ReplyResult:
    """Store a concise destination summary generated by planner_agent."""
    context_variables["destination_features"] = (destination_features or "").strip()
    return ReplyResult(
        message="Destination features stored.",
        context_variables=context_variables,
        target=StayTarget(),
    )


def set_destination(
    state: str = "",
    city: str = "",
    context_variables: Optional[ContextVariables] = None,
) -> ReplyResult:
    """Record destination before handing off to poi_agent.

    Provide **at least one** of *state* or *city* (leave the other empty).
    - State only: attractions are retrieved statewide.
    - City only: state is inferred from meta-data (city→state index).
    - Both: used as given (state must match known US state names).
    """
    if context_variables is None:
        raise TypeError("context_variables is required for set_destination")

    state = (state or "").strip()
    city = (city or "").strip()

    if not state and not city:
        return ReplyResult(
            message=(
                "Need at least a US state name or a city name. "
                "Call set_destination with state=..., city=..., using empty string for the unknown field."
            ),
            context_variables=context_variables,
            target=StayTarget(),
        )

    canonical_state: Optional[str] = None

    if state:
        canonical_state = _available_states.get(state.lower())
        if not canonical_state:
            return ReplyResult(
                message=(
                    f"State '{state}' is not in our database. "
                    f"Examples: {', '.join(sorted(_available_states.values())[:8])} ..."
                ),
                context_variables=context_variables,
                target=StayTarget(),
            )

    if city and not canonical_state:
        canonical_state = _resolve_state_from_city(city)
        if not canonical_state:
            return ReplyResult(
                message=(
                    f"Could not map city '{city}' to a state in our meta-data. "
                    "Ask the customer for the US state or a more specific city name."
                ),
                context_variables=context_variables,
                target=StayTarget(),
            )

    context_variables["destination_state"] = canonical_state
    context_variables["destination_city"] = city if city else None
    _reset_planning_context(context_variables)

    if city:
        msg = f"Destination set to city={city}, state={canonical_state}."
    else:
        msg = f"Destination set to state={canonical_state} (statewide attraction search)."
    return ReplyResult(
        message=msg,
        context_variables=context_variables,
        target=StayTarget(),
    )


# ---- Itinerary tools ----

def mark_itinerary_as_complete(
    final_itinerary: str, context_variables: ContextVariables
) -> ReplyResult:
    """Mark the latest critiqued structured itinerary as accepted by the customer."""
    draft_itinerary = context_variables.get("draft_itinerary")
    if not draft_itinerary:
        return ReplyResult(
            message=(
                "No structured itinerary draft is available. Call submit_itinerary_for_critique "
                "with a valid days/events payload before completing the itinerary."
            ),
            context_variables=context_variables,
            target=AgentNameTarget("planner_agent"),
        )
    try:
        parsed = _normalize_itinerary_payload(json.loads(draft_itinerary))
        itinerary = Itinerary.model_validate(parsed)
    except Exception as exc:
        return ReplyResult(
            message=f"Latest itinerary draft is not valid structured JSON: {exc}",
            context_variables=context_variables,
            target=AgentNameTarget("planner_agent"),
        )
    context_variables["itinerary_confirmed"] = True
    context_variables["structured_itinerary"] = json.dumps(
        itinerary.model_dump(),
        ensure_ascii=False,
    )
    context_variables.pop("timed_itinerary", None)
    context_variables["timed_itinerary_validation"] = None
    context_variables["final_itinerary_presented_to_user"] = False
    context_variables["timed_itinerary_json_for_planner"] = ""
    return ReplyResult(
        message="Itinerary recorded, confirmed, and ready for route timing.",
        context_variables=context_variables,
        target=AgentNameTarget("route_timing_agent"),
    )


def acknowledge_final_itinerary_presented(context_variables: ContextVariables) -> ReplyResult:
    """Mark that planner_agent already delivered the timed itinerary to the customer in prose (end chat)."""
    context_variables["final_itinerary_presented_to_user"] = True
    context_variables["timed_itinerary_json_for_planner"] = ""
    return ReplyResult(
        message="Marked final itinerary as presented to the customer.",
        context_variables=context_variables,
        target=StayTarget(),
    )


def _normalize_draft_for_compare(value: Any) -> str:
    """Canonicalize itinerary payloads so repeated critique loops can be detected."""
    if value is None:
        return ""
    if isinstance(value, str):
        stripped = value.strip()
        try:
            value = json.loads(stripped)
        except Exception:
            return re.sub(r"\s+", " ", stripped)
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    except Exception:
        return re.sub(r"\s+", " ", str(value).strip())


def _format_critic_feedback_message(report: "CriticReport") -> str:
    """Expose concrete critic findings so planner_agent has actionable feedback."""
    lines = [
        f"Critic status: {report.status}.",
        report.summary,
        (
            "Unmet constraint counts: "
            f"hard={report.hard_constraint_count}, soft={report.soft_constraint_count}."
        ),
    ]
    if report.hard_constraints:
        lines.append("Hard constraints:")
        lines.extend(f"- {item}" for item in report.hard_constraints[:6])
    if report.soft_constraints:
        lines.append("Soft constraints:")
        lines.extend(f"- {item}" for item in report.soft_constraints[:6])
    if report.issues:
        lines.append("Issues:")
        for issue in report.issues[:8]:
            pois = ", ".join(issue.related_pois) if issue.related_pois else "n/a"
            lines.append(
                f"- [{issue.severity}] {issue.issue_type} | POIs: {pois} | "
                f"Reason: {issue.conflict_reason} | Fix: {issue.suggested_fix}"
            )
    return "\n".join(lines)


def _format_timed_validation_message(report: Dict[str, Any]) -> str:
    lines = [
        f"Timed itinerary validation status: {report.get('status', 'unknown')}.",
        str(report.get("summary", "")),
        (
            "Execution constraint counts: "
            f"hard={int(report.get('hard_constraint_count') or 0)}, "
            f"soft={int(report.get('soft_constraint_count') or 0)}."
        ),
    ]
    hard_constraints = report.get("hard_constraints") or []
    soft_constraints = report.get("soft_constraints") or []
    issues = report.get("issues") or []
    if hard_constraints:
        lines.append("Execution hard constraints:")
        lines.extend(f"- {item}" for item in hard_constraints[:6])
    if soft_constraints:
        lines.append("Execution soft constraints:")
        lines.extend(f"- {item}" for item in soft_constraints[:6])
    if issues:
        lines.append("Execution issues:")
        for issue in issues[:8]:
            pois = ", ".join(issue.get("related_pois") or []) or "n/a"
            lines.append(
                f"- [{issue.get('severity', 'unknown')}] {issue.get('issue_type', 'unknown')} | "
                f"POIs: {pois} | Reason: {issue.get('conflict_reason', '')} | "
                f"Fix: {issue.get('suggested_fix', '')}"
            )
    return "\n".join(lines)


def _normalize_itinerary_payload(payload: Any) -> Any:
    """Repair common schema drift from structured tool payloads before validation."""
    if not isinstance(payload, dict):
        return payload
    days = payload.get("days")
    if not isinstance(days, list):
        return payload

    for day in days:
        if not isinstance(day, dict):
            continue
        events = day.get("events")
        if not isinstance(events, list):
            continue
        for event in events:
            if not isinstance(event, dict):
                continue
            event_type = str(event.get("type") or "").strip()
            if event_type != "Travel":
                event["type"] = "Attraction"
    return payload


def restart_trip_planning(context_variables: ContextVariables) -> ReplyResult:
    """Full restart after the customer rejects or wants to redo the plan post-itinerary."""
    critic_feedback = context_variables.get("critic_feedback") or {}
    user_feedback = (context_variables.get("last_user_feedback") or "").strip().lower()
    explicit_restart = any(
        token in user_feedback
        for token in ("restart", "start over", "redo", "replan", "different destination", "another destination")
    )
    if critic_feedback.get("status") == "revise" and not explicit_restart:
        return ReplyResult(
            message=(
                "Do not restart the whole trip for a revise critique. Repair the listed issues locally "
                "using the current POI candidate pool. Only restart if the customer explicitly asks for "
                "a full redo or a different destination."
            ),
            context_variables=context_variables,
            target=AgentNameTarget("planner_agent"),
        )
    restart_attempt_count = int(context_variables.get("restart_attempt_count") or 0)
    if restart_attempt_count >= MAX_RESTART_ATTEMPTS:
        context_variables["critic_force_exit"] = "degrade"
        context_variables["planner_degraded"] = True
        return ReplyResult(
            message=(
                f"Trip planning restart has already been attempted {MAX_RESTART_ATTEMPTS} times. "
                "Do not restart again. Produce a degraded final plan using the current candidate pool."
            ),
            context_variables=context_variables,
            target=AgentNameTarget("planner_agent"),
        )
    _reset_planning_context(context_variables)
    context_variables["itinerary_confirmed"] = False
    context_variables["structured_itinerary"] = None
    context_variables["restart_attempt_count"] = restart_attempt_count + 1
    return ReplyResult(
        message=(
            "Planning reset: call set_destination if needed, then set_destination_features, "
            "queue_preference_retrieval, hand off to pref_agent, set_poi_preference_summary, "
            "and hand off to poi_agent when poi_handoff_needed is true."
        ),
        context_variables=context_variables,
        target=StayTarget(),
    )


class CriticIssue(BaseModel):
    issue_type: str
    related_pois: List[str] = Field(default_factory=list)
    conflict_reason: str
    suggested_fix: str
    severity: str = "medium"


class CriticReport(BaseModel):
    status: str
    summary: str
    hard_constraint_count: int = 0
    soft_constraint_count: int = 0
    hard_constraints: List[str] = Field(default_factory=list)
    soft_constraints: List[str] = Field(default_factory=list)
    issues: List[CriticIssue] = Field(default_factory=list)


def submit_itinerary_for_critique(
    days: List[Day], context_variables: ContextVariables
) -> ReplyResult:
    """Store a structured itinerary draft and send it to deterministic critique.

    This tool is intentionally schema-shaped for LLM tool calling: the planner must submit a
    top-level days/events payload rather than free-form prose or a JSON string.
    """
    critic_feedback = context_variables.get("critic_feedback") or {}
    critic_status = critic_feedback.get("status")
    force_exit = (context_variables.get("critic_force_exit") or "").strip()
    critic_iteration_count = int(context_variables.get("critic_iteration_count") or 0)
    try:
        itinerary = Itinerary.model_validate(
            _normalize_itinerary_payload({"days": _model_to_dict(days)})
        )
    except Exception as exc:
        return ReplyResult(
            message=(
                "Structured itinerary payload is invalid. Call submit_itinerary_for_critique "
                f"again with days[].events[] matching the Itinerary schema. Error: {exc}"
            ),
            context_variables=context_variables,
            target=AgentNameTarget("planner_agent"),
        )
    draft_itinerary = json.dumps(itinerary.model_dump(), ensure_ascii=False)
    normalized_draft = _normalize_draft_for_compare(draft_itinerary)
    last_critic_draft = context_variables.get("last_critic_draft") or ""

    if critic_status == "pass" and normalized_draft and normalized_draft == last_critic_draft:
        return ReplyResult(
            message=(
                "This exact itinerary draft already passed critic checks. Do not re-submit it for "
                "critique. Enter completion flow now and call mark_itinerary_as_complete."
            ),
            context_variables=context_variables,
            target=AgentNameTarget("planner_agent"),
        )
    if force_exit == "degrade" and normalized_draft and normalized_draft == last_critic_draft:
        return ReplyResult(
            message=(
                "Critique loop detected for an unchanged itinerary draft. Do not re-submit it for "
                "critique. Produce a hard-constraints-first degraded final plan and then call "
                "mark_itinerary_as_complete."
            ),
            context_variables=context_variables,
            target=AgentNameTarget("planner_agent"),
        )
    if critic_iteration_count >= MAX_CRITIC_ITERATIONS and critic_status != "pass":
        context_variables["critic_force_exit"] = "degrade"
        context_variables["planner_degraded"] = True
        return ReplyResult(
            message=(
                f"Critique attempts already reached the limit ({MAX_CRITIC_ITERATIONS}). "
                "Do not submit another critique. Produce the best hard-constraints-first degraded plan "
                "and call mark_itinerary_as_complete."
            ),
            context_variables=context_variables,
            target=AgentNameTarget("planner_agent"),
        )

    context_variables["draft_itinerary_text"] = ""
    context_variables["draft_itinerary"] = draft_itinerary
    context_variables["structured_itinerary"] = None
    return ReplyResult(
        message="Draft itinerary stored for critique.",
        context_variables=context_variables,
        target=AgentNameTarget("critic_agent"),
    )


def store_critic_feedback(
    critique_report: str, context_variables: ContextVariables
) -> ReplyResult:
    """Persist critic feedback into context for planner_agent to revise locally."""
    report = CriticReport.model_validate(json.loads(critique_report))
    report_data = report.model_dump()
    current_draft_signature = _normalize_draft_for_compare(context_variables.get("draft_itinerary"))
    current_feedback_signature = _normalize_draft_for_compare(report_data)
    previous_draft_signature = context_variables.get("last_critic_draft") or ""
    previous_feedback_signature = context_variables.get("last_critic_feedback_signature") or ""

    same_draft = current_draft_signature and current_draft_signature == previous_draft_signature
    same_feedback = (
        current_feedback_signature
        and current_feedback_signature == previous_feedback_signature
    )

    if report.status == "pass":
        context_variables["critic_stall_count"] = 0
        context_variables["critic_force_exit"] = "pass"
    elif same_draft and same_feedback:
        context_variables["critic_stall_count"] = int(
            context_variables.get("critic_stall_count") or 0
        ) + 1
        report.status = "degrade"
        report.summary = (
            "Critic feedback is unchanged for an unchanged itinerary draft. Stop revising this "
            "draft in a loop and produce a hard-constraints-first degraded final plan."
        )
        report.issues.append(
            CriticIssue(
                issue_type="critique_loop_detected",
                related_pois=[],
                conflict_reason=(
                    "The latest itinerary draft is materially unchanged and the critic feedback "
                    "did not change either."
                ),
                suggested_fix=(
                    "Do not re-submit the same draft. Either finalize if acceptable or produce a "
                    "degraded executable itinerary that clearly states the compromise."
                ),
                severity="hard",
            )
        )
        context_variables["critic_force_exit"] = "degrade"
        context_variables["planner_degraded"] = True
    else:
        context_variables["critic_stall_count"] = 0
        context_variables["critic_force_exit"] = ""

    report_data = report.model_dump()
    context_variables["critic_feedback"] = report_data
    context_variables["critic_iteration_count"] = (
        int(context_variables.get("critic_iteration_count") or 0) + 1
    )
    if (
        int(context_variables.get("critic_iteration_count") or 0) >= MAX_CRITIC_ITERATIONS
        and report.status != "pass"
    ):
        report.status = "degrade"
        report.summary = (
            f"Critique reached the maximum retry limit ({MAX_CRITIC_ITERATIONS}). "
            "Stop revising and produce the best executable degraded plan."
        )
        report_data = report.model_dump()
        context_variables["critic_feedback"] = report_data
        context_variables["planner_degraded"] = True
        context_variables["critic_force_exit"] = "degrade"
    if report.status == "degrade":
        context_variables["planner_degraded"] = True
        context_variables["critic_force_exit"] = "degrade"
    context_variables["last_critic_draft"] = current_draft_signature
    context_variables["last_critic_feedback_signature"] = _normalize_draft_for_compare(report_data)
    return ReplyResult(
        message=_format_critic_feedback_message(report),
        context_variables=context_variables,
        target=AgentNameTarget("planner_agent"),
    )


def _planner_sync_user_feedback(agent: ConversableAgent, messages: List[Dict[str, Any]]) -> None:
    """Fill last_user_feedback / last_user_feedback_empty from the latest customer message."""
    cv = getattr(agent, "context_variables", None)
    if cv is None:
        return
    msg_list: List[Dict[str, Any]] = list(messages or [])
    mgr = getattr(agent, "_group_manager", None)
    if mgr is not None and getattr(mgr, "groupchat", None) is not None:
        gm = mgr.groupchat.messages
        if isinstance(gm, list) and len(gm) > 0:
            msg_list = gm
    last_customer = ""
    for msg in reversed(msg_list):
        name = msg.get("name")
        role = msg.get("role")
        if name == "customer" or (role == "user" and name in (None, "customer")):
            content = msg.get("content")
            if isinstance(content, list):
                parts = []
                for p in content:
                    if isinstance(p, dict) and p.get("type") == "text":
                        parts.append(p.get("text", ""))
                last_customer = "".join(parts)
            else:
                last_customer = content if isinstance(content, str) else ""
            break
    stripped = (last_customer or "").strip()
    cv["last_user_feedback"] = stripped
    cv["last_user_feedback_empty"] = not bool(stripped)


_TIMED_ITINERARY_JSON_PLANNER_CAP = 32000


def _planner_sync_timed_itinerary_digest(agent: ConversableAgent, messages: List[Dict[str, Any]]) -> None:
    """Expose timed_itinerary JSON for the planner template when NL delivery is still pending."""
    cv = getattr(agent, "context_variables", None)
    if cv is None:
        return
    if cv.get("final_itinerary_presented_to_user"):
        cv["timed_itinerary_json_for_planner"] = ""
        return
    timed = cv.get("timed_itinerary")
    if not timed:
        cv["timed_itinerary_json_for_planner"] = ""
        return
    validation = cv.get("timed_itinerary_validation") or {}
    if validation.get("status") != "pass":
        cv["timed_itinerary_json_for_planner"] = ""
        return
    try:
        blob = json.dumps(timed, ensure_ascii=False)
    except (TypeError, ValueError):
        cv["timed_itinerary_json_for_planner"] = ""
        return
    if len(blob) > _TIMED_ITINERARY_JSON_PLANNER_CAP:
        blob = (
            blob[: _TIMED_ITINERARY_JSON_PLANNER_CAP]
            + "\n... [truncated for prompt size; use thread context if needed]"
        )
    cv["timed_itinerary_json_for_planner"] = blob


def _critic_agent_reply(
    recipient: ConversableAgent,
    messages=None,
    sender=None,
    config=None,
) -> tuple:
    cv = getattr(recipient, "context_variables", None)
    if cv is None:
        return True, "No context variables available for critique."
    report = evaluate_itinerary(
        cv.get("draft_itinerary"),
        cv.get("poi_research_results"),
        cv.get("poi_candidates"),
        max_iteration=MAX_CRITIC_ITERATIONS,
        iteration_count=int(cv.get("critic_iteration_count") or 0),
    )
    critique_json = json.dumps(report, ensure_ascii=False)
    result = store_critic_feedback(critique_json, cv)
    return True, result.message


# =====================================================================
# 5. Create Agents
# =====================================================================

PLANNER_SYSTEM_TEMPLATE = (
    "You are a trip planner agent.\n\n"
    "MANDATORY BEFORE PLANNING:\n"
    "- You MUST ask the customer for their DESTINATION (state and/or city) AND the "
    "NUMBER OF DAYS they want to travel.  Do NOT draft or propose any itinerary "
    "until BOTH pieces of information have been explicitly confirmed by the customer.\n\n"
    "IMPORTANT LOCATION RULES:\n"
    "- Location can be a US STATE only, a CITY only, or BOTH.\n"
    "- Call set_destination with state=... and/or city=...; use empty string \"\" "
    "for the field you do not have.\n"
    "- Examples: state only → set_destination(state=\"California\", city=\"\"). "
    "City only → set_destination(state=\"\", city=\"Los Angeles\").\n"
    "- If the customer gives only a state, do NOT insist on a city.\n\n"
    "POI AGENT HANDOFF (important):\n"
    "- The handoff to poi_agent is only offered when poi_handoff_needed is true "
    "(after you call set_poi_preference_summary following pref retrieval). After poi_agent returns, "
    "that flag becomes false — do NOT hand off again until the destination changes (call "
    "set_destination again) or the customer needs a full replan (restart_trip_planning).\n\n"
    "FINAL ITINERARY DELIVERY (only you talk to the customer here):\n"
    "- After you call mark_itinerary_as_complete, route_timing_agent fills timed_itinerary (Travel legs) "
    "and immediately returns to you. Do not treat route_timing_agent as user-facing.\n"
    "- route_timing_agent also runs a final execution check over timed_itinerary and stores the result in "
    "timed_itinerary_validation.\n"
    "- When timed_itinerary_json_for_planner below is non-empty, convert that structured plan into a "
    "clear, friendly natural-language itinerary for the customer (day-by-day; include movement/transit "
    "where present) only if timed_itinerary_validation.status is 'pass'. In the same turn, call "
    "acknowledge_final_itinerary_presented() after your message so the session can end.\n"
    "- Apart from pref_agent (memory retrieval on its first turn), you are the only agent that should "
    "address the customer in natural language for planning outcomes.\n\n"
    "WORKFLOW:\n"
    "1. Ask the customer for DESTINATION and NUMBER OF DAYS if missing.\n"
    "2. Call set_destination as soon as you have at least a state OR a city.\n"
    "3. Immediately after set_destination, generate a short destination summary describing the "
    "travel character of that city/state, then call set_destination_features.\n"
    "4. Write ONE paragraph that merges (a) the customer's confirmed trip needs (days, pace, "
    "interests, constraints) and (b) the destination character from destination_features. "
    "Call queue_preference_retrieval(query=that paragraph) so pref_agent can run mem0 semantic search.\n"
    "5. When preference_retrieval_needed is true, hand off to pref_agent once. After it returns, "
    "read USER PREFERENCES below (raw retrieval), distill a short summary for POI scoring, and "
    "call set_poi_preference_summary(summary=...).\n"
    "6. When poi_handoff_needed is true, hand off to poi_agent once to get the researched "
    "candidate list for the current destination.\n"
    "7. Present a concise shortlist of POIs to the customer (from that researched list only).\n"
    "8. FEEDBACK after the POI shortlist:\n"
    "   - If last_user_feedback_empty is True (customer sent an empty message / skip): treat as "
    "\"no change\" and immediately proceed to build the day-by-day itinerary using ONLY POI names "
    "that appear in the researched POI Markdown already in this thread. Do NOT call poi_agent again.\n"
    "   - If last_user_feedback_empty is False: interpret last_user_feedback and adjust your "
    "selected POIs using ONLY names from that same POI candidate list (no invented venues).\n"
    "9. Build a diverse day-by-day itinerary using ONLY those attractions. Use the POI research "
    "fields in context to vary categories, visit intensity, and itinerary roles across days. "
    "Each event MUST have type='Attraction', 'location', 'city', and 'description'.\n"
    "10. Before presenting a final itinerary, you MUST call submit_itinerary_for_critique via tool "
    "calling with a structured days/events payload matching the Itinerary schema. Do not pass "
    "natural language, Markdown, or a JSON string as the draft. The critic feedback is stored in "
    "context_variables['critic_feedback'].\n"
    "11. If critic_feedback.status is 'pass', do NOT submit the same draft again. Immediately move "
    "into completion flow and call mark_itinerary_as_complete.\n"
    "12. After route_timing_agent runs: if timed_itinerary_validation.status is 'pass', narrate the "
    "timed plan to the customer and call acknowledge_final_itinerary_presented(). If it is not 'pass', "
    "do NOT show the timed itinerary to the customer. Repair the listed execution issues in the current "
    "itinerary draft and submit the revised draft for critique again.\n"
    "13. If critic_feedback.status is 'revise', keep the already-valid portions and perform local "
    "repairs only for the listed issues. Do not rewrite everything from scratch.\n"
    "14. The critic feedback issues are authoritative. You MUST inspect "
    "critic_feedback.issues and repair the concrete POIs/constraints listed there before sending "
    "another draft.\n"
    "15. If critic_feedback.status is 'degrade', critic_force_exit is 'degrade', or the latest "
    "draft is unchanged while critic feedback is unchanged, stop the loop and produce a "
    "hard-constraints-first degraded itinerary and clearly mark the compromise in the itinerary description.\n"
    "16. NEVER call restart_trip_planning just because critic_feedback.status is 'revise'. In that case "
    "you must repair only the specific issues in critic_feedback.issues using the current researched "
    "POI pool.\n"
    "17. FEEDBACK after you show the draft ITINERARY:\n"
    "   - If last_user_feedback_empty is True: treat as approval to continue — call "
    "mark_itinerary_as_complete with a short summary of the agreed plan.\n"
    "   - If last_user_feedback_empty is False: if the customer wants a full redo, call "
    "restart_trip_planning, then set_destination if the destination changes, and only hand off to "
    "poi_agent when poi_handoff_needed is true again. For minor edits, revise the itinerary "
    "using the same candidate POI pool without re-querying the graph unless the destination changed.\n"
    "18. When the customer expresses a new preference during chat, call queue_user_preference_storage. "
    "That only queues the write; pref_agent will persist it.\n\n"
    "CONTEXT (updated every turn before you speak):\n"
    "- final_itinerary_presented_to_user: {final_itinerary_presented_to_user}\n"
    "- timed_itinerary_json_for_planner: {timed_itinerary_json_for_planner}\n"
    "- preference_retrieval_needed: {preference_retrieval_needed}\n"
    "- poi_handoff_needed: {poi_handoff_needed}\n"
    "- user_preferences_for_poi: {user_preferences_for_poi}\n"
    "- last_user_feedback_empty: {last_user_feedback_empty}\n"
    "- last_user_feedback: {last_user_feedback}\n"
    "- destination_features: {destination_features}\n"
    "- critic_feedback: {critic_feedback}\n"
    "- critic_iteration_count: {critic_iteration_count}\n"
    "- critic_stall_count: {critic_stall_count}\n"
    "- critic_force_exit: {critic_force_exit}\n"
    "- timed_itinerary_validation: {timed_itinerary_validation}\n"
    "- planner_degraded: {planner_degraded}\n\n"
    "USER PREFERENCES (raw semantic hits from pref_agent / mem0.search):\n{user_preferences}"
)

planner_agent = ConversableAgent(
    name="planner_agent",
    system_message=PLANNER_SYSTEM_TEMPLATE,
    llm_config=agent_llm_config,
    functions=[
        mark_itinerary_as_complete,
        acknowledge_final_itinerary_presented,
        queue_user_preference_storage,
        queue_preference_retrieval,
        set_poi_preference_summary,
        set_destination,
        set_destination_features,
        restart_trip_planning,
        submit_itinerary_for_critique,
    ],
    update_agent_state_before_reply=[
        _planner_sync_user_feedback,
        _planner_sync_timed_itinerary_digest,
        UpdateSystemMessage(PLANNER_SYSTEM_TEMPLATE),
    ],
)

poi_agent = ConversableAgent(
    name="poi_agent",
    system_message=(
        "Return candidate attractions for the requested location (city and/or state) as **Markdown**. "
        "Each **primary** POI (preference category match) is followed by **Nearby (NEAR graph)** POIs "
        "that are neighbours of that primary — use only names from this document as-is."
    ),
    llm_config=False,
)


# ---------------------------------------------------------------------------
# pref_agent: optional mem0.add (queued storage), or mem0.search for semantic
# preference retrieval. Writes raw hits to user_preferences. No LLM call.
# ---------------------------------------------------------------------------

def _pref_agent_reply(
    recipient: ConversableAgent,
    messages=None,
    sender=None,
    config=None,
) -> tuple:
    cv = getattr(recipient, "context_variables", None)
    did_store = False
    if cv is not None and cv.get("preference_storage_needed") and cv.get("pending_preference"):
        preference_text = cv["pending_preference"]
        mem0_client.add(
            preference_text,
            user_id=USER_ID,
            metadata={
                "agent_id": "pref_agent",
                "actor_id": "planner_agent",
                "role": "assistant",
                "source": "conversation_preference",
                "confidence": 0.75,
                "timestamp": int(time.time()),
            },
        )
        cv["preference_storage_needed"] = False
        cv["pending_preference"] = ""
        did_store = True

    retrieval_q = (cv.get("preference_retrieval_query") or "").strip() if cv is not None else ""
    if cv is not None and cv.get("preference_retrieval_needed") and retrieval_q:
        cv["user_preferences_for_poi"] = ""
        try:
            raw = mem0_client.search(
                retrieval_q,
                user_id=USER_ID,
                limit=PREF_SEMANTIC_SEARCH_LIMIT,
                rerank=True,
            )
        except Exception as exc:
            logger.exception("mem0.search failed: %s", exc)
            prefs_text = f"Preference semantic search failed: {exc}"
            if cv is not None:
                cv["user_preferences"] = prefs_text
                cv["preference_retrieval_needed"] = False
                cv["preference_retrieval_query"] = ""
            return True, prefs_text

        memories = raw.get("results", []) if isinstance(raw, dict) else (raw or [])
        lines = []
        for i, m in enumerate(memories, 1):
            score = m.get("score", "")
            mem = m.get("memory", "")
            lines.append(f"{i}. (score={score}) {mem}")
        prefs_text = (
            f"Semantic retrieval query:\n{retrieval_q}\n\n"
            + (
                "Matching memories:\n" + "\n".join(lines)
                if lines
                else "No matching memories above threshold."
            )
        )
        cv["user_preferences"] = prefs_text
        cv["preference_retrieval_needed"] = False
        cv["preference_retrieval_query"] = ""
        return True, prefs_text

    if did_store:
        return True, "Preference stored in mem0."

    if cv is not None and cv.get("preference_storage_needed"):
        # Queued but empty pending_preference — nothing to persist.
        cv["preference_storage_needed"] = False
        cv["pending_preference"] = ""

    noop_msg = (
        "No preference storage or semantic retrieval was queued. "
        "Planner should call queue_preference_retrieval after set_destination_features "
        "(or queue_user_preference_storage for a new memory)."
    )
    return True, noop_msg


pref_agent = ConversableAgent(
    name="pref_agent",
    system_message="Retrieve user travel preferences from the vector database.",
    llm_config=False,
)
pref_agent.register_reply(
    [ConversableAgent, None],
    _pref_agent_reply,
    position=0,
    remove_other_reply_funcs=True,
)

critic_agent = ConversableAgent(
    name="critic_agent",
    system_message="Apply deterministic itinerary rules and store structured critique feedback.",
    llm_config=False,
)
critic_agent.register_reply(
    [ConversableAgent, None],
    _critic_agent_reply,
    position=0,
    remove_other_reply_funcs=True,
)

route_timing_agent = ConversableAgent(
    name="route_timing_agent",
    system_message=(
        "Internal agent: compute Travel segments only; output is not shown to the end user."
    ),
    llm_config=False,
)


def _route_timing_agent_reply(
    recipient: ConversableAgent,
    messages=None,
    sender=None,
    config=None,
) -> tuple:
    cv = getattr(recipient, "context_variables", None)
    if cv is None:
        return True, "No context variables for route timing."
    rr = update_itinerary_with_travel_times(cv)
    logger.debug("route_timing_agent: %s", rr.message)
    timed_validation = validate_timed_itinerary(
        cv.get("timed_itinerary"),
        cv.get("poi_research_results"),
        cv.get("poi_candidates"),
    )
    cv["timed_itinerary_validation"] = timed_validation
    if timed_validation.get("status") != "pass":
        cv["timed_itinerary_json_for_planner"] = ""
    return True, _format_timed_validation_message(timed_validation)


route_timing_agent.register_reply(
    [ConversableAgent, None],
    _route_timing_agent_reply,
    position=0,
    remove_other_reply_funcs=True,
)


# =====================================================================
# 6. Custom reply for poi_agent
# =====================================================================

def _extract_destination_from_context(
    recipient: ConversableAgent,
    messages: List[Dict[str, Any]],
) -> tuple:
    """Return (state, city) from ContextVariables; city may be None for statewide.
    Falls back to scanning messages for known state names."""
    cv = getattr(recipient, "context_variables", None)
    state = None
    city = None
    if cv and hasattr(cv, "get"):
        state = cv.get("destination_state")
        city = cv.get("destination_city")

    if city and not state:
        state = _resolve_state_from_city(str(city))

    if state:
        return state, city

    for msg in reversed(messages):
        content = msg.get("content", "")
        if not content:
            continue
        content_lower = content.lower()
        for key, canonical in _available_states.items():
            if key in content_lower:
                return canonical, city
    return None, None


MAX_CANDIDATE_ATTRACTIONS = 50


def _md_inline(s: Any) -> str:
    """Single-line safe text for Markdown (no newlines)."""
    if s is None:
        return ""
    return str(s).replace("\n", " ").strip()


def _format_primary_block_md(idx: int, a: Dict[str, Any]) -> List[str]:
    """Markdown lines for one primary (match-category) POI."""
    name = _md_inline(a.get("name")) or "Unknown"
    rating = a.get("avg_rating")
    reviews = a.get("num_of_reviews")
    cat = a.get("matched_category")
    sim = a.get("similarity_score")
    lines = [
        f"## {idx}. {name}",
        "",
        f"- **Rating:** {rating if rating is not None else '—'} · **Reviews:** {reviews if reviews is not None else '—'}",
    ]
    if cat is not None:
        extra = f" (preference match score {sim:.3f})" if isinstance(sim, (int, float)) else ""
        lines.append(f"- **Matched category:** {_md_inline(cat)}{extra}")
    lines.append("")
    return lines


def _format_near_block_md(neighbors: List[Dict[str, Any]]) -> List[str]:
    """Markdown for NEAR expansions under one primary."""
    if not neighbors:
        return []
    lines = [
        "#### Nearby (NEAR graph)",
        "",
        "These POIs are linked by a **NEAR** edge to the primary above (same city/state query). "
        "Use them only as optional add-ons tied to that primary.",
        "",
    ]
    for n in neighbors:
        nm = _md_inline(n.get("name")) or "Unknown"
        rating = n.get("avg_rating")
        rev = n.get("num_of_reviews")
        lines.append(
            f"- **{nm}** — rating {rating if rating is not None else '—'}, "
            f"reviews {rev if rev is not None else '—'}"
        )
    lines.append("")
    return lines


def _format_attraction_list(
    direct: List[Dict],
    near: List[Dict],
    city: Optional[str],
    state: str,
) -> str:
    """Return Markdown for planner_agent: primary POIs (match category), each followed by NEAR neighbours.

    At most MAX_CANDIDATE_ATTRACTIONS distinct attraction names total (primaries first, then
    neighbours attached in order; each neighbour appears at most once, under the first primary
    that claims it in ``near_of``).
    """
    if not direct:
        loc = f"city '{city}'" if city else f"state '{state}'"
        return f"No matching attractions found for {loc}."

    loc_label = f"in {city}, {state}" if city else f"statewide in {state}"
    direct_names = {d.get("name") for d in direct if d.get("name")}

    seen: set = set()
    blocks: List[tuple[Dict[str, Any], List[Dict[str, Any]]]] = []
    primary_idx = 0

    for d in direct:
        if len(seen) >= MAX_CANDIDATE_ATTRACTIONS:
            break
        dname = d.get("name")
        if not dname or dname in seen:
            continue
        seen.add(dname)
        primary_idx += 1

        neighbors: List[Dict[str, Any]] = []
        for n in near:
            if len(seen) >= MAX_CANDIDATE_ATTRACTIONS:
                break
            nname = n.get("name")
            if not nname or nname in seen:
                continue
            if nname in direct_names:
                continue
            near_of = n.get("near_of") or []
            if isinstance(near_of, str):
                near_of = [near_of]
            if dname not in near_of:
                continue
            neighbors.append(n)
            seen.add(nname)

        blocks.append((d, neighbors))

    header = "\n".join(
        [
            f"# Candidate attractions ({loc_label})",
            "",
            "**Primary** POIs are those that **match user preference categories** (when preferences exist) "
            "or **all attractions in scope** when no category match is used.",
            "",
            "Under each primary, **Nearby (NEAR graph)** lists POIs reached only via a **NEAR** relationship "
            "to that primary — treat them as geographically related options for that primary.",
            "",
            f"At most **{MAX_CANDIDATE_ATTRACTIONS}** distinct names total (primaries + neighbours).",
            "",
            "---",
            "",
        ]
    )

    parts: List[str] = [header]
    run_idx = 0
    for d, neighbors in blocks:
        run_idx += 1
        parts.extend(_format_primary_block_md(run_idx, d))
        parts.extend(_format_near_block_md(neighbors))

    return "\n".join(parts).rstrip() + "\n"


def _build_research_input(scored_attractions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    poi_list = []
    for idx, item in enumerate(scored_attractions, 1):
        categories = item.get("categories") or []
        primary_category = categories[0] if categories else "Attraction"
        poi_list.append(
            {
                "poi_id": f"poi_{idx}",
                "place_name": item.get("name") or "Unknown",
                "category": primary_category,
                "city": item.get("city"),
                "region": item.get("state"),
            }
        )
    return poi_list


def _format_research_markdown(
    scored_attractions: List[Dict[str, Any]],
    researched: List[POIStructuredInfo],
    city: Optional[str],
    state: str,
) -> str:
    if not scored_attractions:
        loc = f"{city}, {state}" if city else state
        return f"No researched attractions found for {loc}."

    research_by_name = {item.place_name: item for item in researched}
    loc_label = f"{city}, {state}" if city else state
    lines = [
        f"# Researched candidate attractions ({loc_label})",
        "",
        "Use only POI names from this list. Scores combine user preference memory and destination features.",
        "",
    ]
    for idx, item in enumerate(scored_attractions, 1):
        info = research_by_name.get(item.get("name", ""))
        categories = ", ".join(item.get("categories") or []) or "Attraction"
        lines.extend(
            [
                f"## {idx}. {item.get('name', 'Unknown')}",
                "",
                f"- **Score:** {item.get('score', 0):.3f}",
                f"- **Category:** {categories}",
                f"- **City:** {_md_inline(item.get('city')) or 'Unknown'}",
                f"- **Best visit time:** {info.best_visit_time if info else 'unknown'}",
                f"- **Recommended duration:** {info.recommended_duration if info else 'unknown'}",
                f"- **Opening hours:** {info.opening_hours if info else 'unknown'}",
                f"- **Crowd level:** {info.crowd_level if info else 'unknown'}",
                f"- **Itinerary role:** {info.itinerary_role if info else 'unknown'}",
            ]
        )
        if info and info.notes:
            lines.append(f"- **Notes:** {'; '.join(info.notes[:2])}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def poi_agent_reply(
    recipient: ConversableAgent,
    messages: Optional[List[Dict[str, Any]]] = None,
    sender: Optional[Any] = None,
    config: Optional[Any] = None,
) -> tuple:
    """Custom reply function:
    state+city extraction → check Neo4j for State node → build graph if missing
    → preference-based category match → Cypher query → formatted output."""

    if messages is None:
        return True, "No messages received."

    # Step 1: Identify state and city (city optional → statewide query)
    state, city = _extract_destination_from_context(recipient, messages)
    if not state:
        return True, (
            "I could not determine the destination. "
            "Please ask planner_agent to call set_destination with a US state and/or a city."
        )

    # Validate state against available meta files
    if state.lower() not in _available_states:
        return True, (
            f"State '{state}' has no meta-data file. "
            f"Available states: {', '.join(sorted(_available_states.values())[:10])} ..."
        )

    # Step 2: Check if state graph already exists in Neo4j; build if not
    if not mem0_client.state_exists_in_graph(state):
        logger.info("State '%s' not in Neo4j, building graph from meta-data ...", state)
        graph_data = prepare_state_graph_data(state, META_DIR)
        mem0_client.build_attraction_graph(
            graph_data["state"], graph_data["attractions"], graph_data["near_pairs"]
        )
        logger.info("Graph for '%s' built successfully.", state)
    else:
        logger.info("State '%s' already exists in Neo4j, skipping graph build.", state)

    # Step 3: Get user preferences and destination features from context variables
    prefs = ""
    destination_features = ""
    cv = getattr(recipient, "context_variables", None)
    if cv and hasattr(cv, "get"):
        for_poi = (cv.get("user_preferences_for_poi") or "").strip()
        prefs = for_poi or (cv.get("user_preferences") or "")
        destination_features = cv.get("destination_features", "")

    query_texts = [prefs, destination_features]
    scored_attractions = mem0_client.query_scored_attractions(
        state_name=state,
        city=city,
        query_texts=query_texts,
        limit=MAX_CANDIDATE_ATTRACTIONS,
    )
    logger.info(
        "State=%s, City=%s, scored attractions=%d (top-5 names=%s)",
        state,
        city,
        len(scored_attractions),
        [item.get("name") for item in scored_attractions[:5]],
    )

    scored_attractions = scored_attractions[:8]
    poi_inputs = _build_research_input(scored_attractions)
    researched_infos = process_pois(
        poi_inputs,
        max_workers=min(4, len(poi_inputs) or 1),
        debug=False,
    )

    # Step 6: Format and return
    body = _format_research_markdown(scored_attractions, researched_infos, city, state)
    if cv is not None:
        cv["poi_handoff_needed"] = False
        cv["poi_candidates_markdown"] = body
        cv["poi_candidates"] = scored_attractions
        cv["poi_research_results"] = [item.model_dump() for item in researched_infos]
        cv["poi_research_markdown"] = body
    return True, body


# Register the custom reply for poi_agent.
poi_agent.register_reply(
    [ConversableAgent, None],
    poi_agent_reply,
    position=0,
    remove_other_reply_funcs=True,
)

# =====================================================================
# 7. User proxy
# =====================================================================

customer = UserProxyAgent(name="customer", code_execution_config=False, human_input_mode= "NEVER")

# =====================================================================
# 8. Register handoffs
# =====================================================================

planner_agent.handoffs.add_many(
    [
        OnCondition(
            target=TerminateTarget(),
            condition=StringLLMCondition(
                "Final natural-language itinerary was delivered; end the session."
            ),
            available=StringAvailableCondition("final_itinerary_presented_to_user"),
        ),
        OnCondition(
            target=AgentTarget(pref_agent),
            condition=StringLLMCondition(
                "A queued user preference needs to be persisted to memory by pref_agent."
            ),
            available=StringAvailableCondition("preference_storage_needed"),
        ),
        OnCondition(
            target=AgentTarget(pref_agent),
            condition=StringLLMCondition(
                "Planner queued semantic preference retrieval; pref_agent must run mem0.search "
                "using preference_retrieval_query."
            ),
            available=StringAvailableCondition("preference_retrieval_needed"),
        ),
        OnCondition(
            target=AgentTarget(poi_agent),
            condition=StringLLMCondition(
                "Need candidate attractions from the knowledge graph for the current destination. "
                "Only use when set_destination has already been called and the customer needs a fresh "
                "candidate list (poi_handoff_needed is true)."
            ),
            available=StringAvailableCondition("poi_handoff_needed"),
        ),
    ]
)
planner_agent.handoffs.set_after_work(RevertToUserTarget())

pref_agent.handoffs.set_after_work(AgentTarget(planner_agent))
poi_agent.handoffs.set_after_work(AgentTarget(planner_agent))
critic_agent.handoffs.set_after_work(AgentTarget(planner_agent))
route_timing_agent.handoffs.set_after_work(AgentTarget(planner_agent))


def _normalize_chat_message_content(content: Any) -> str:
    """Flatten AG2/OpenAI-style assistant message content to plain text."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: List[str] = []
        for p in content:
            if isinstance(p, dict) and p.get("type") == "text":
                parts.append(str(p.get("text", "")))
            elif isinstance(p, str):
                parts.append(p)
        return "".join(parts).strip()
    return str(content).strip()


def _last_planner_agent_reply_text(chat_result: Any) -> Optional[str]:
    """Most recent planner_agent message with non-empty visible text (from chat_history)."""
    hist = getattr(chat_result, "chat_history", None)
    if not isinstance(hist, list):
        return None
    for msg in reversed(hist):
        if msg.get("name") != "planner_agent":
            continue
        text = _normalize_chat_message_content(msg.get("content"))
        if text:
            return text
    return None


def print_itinerary(itinerary_data: Dict[str, Any]) -> None:
    width = 80
    icons = {"Travel": "🚶", "Attraction": "🏛️"}

    try:
        city_name = itinerary_data["days"][0]["events"][0]["city"]
    except (KeyError, IndexError):
        city_name = "Unknown"

    print(f"\n{'=' * width}")
    print(f"Itinerary for {city_name}".center(width))
    print(f"{'=' * width}")

    for day_num, day in enumerate(itinerary_data["days"], 1):
        print(f"\nDay {day_num}".center(width))
        print("-" * width)
        for event in day["events"]:
            event_type = event["type"]
            print(f"\n  {icons.get(event_type, '📍')} {event['location']}")
            if event_type != "Travel":
                words = event["description"].split()
                line = "    "
                for word in words:
                    if len(line) + len(word) + 1 <= 76:
                        line += word + " "
                    else:
                        print(line)
                        line = "    " + word + " "
                if line.strip():
                    print(line)
            else:
                print(f"    {event['description']}")
        print("\n" + "-" * width)


def print_last_vector_store_memories(
    memory: Memory,
    user_id: str,
    *,
    count: int = 20,
    fetch_limit: int = 5000,
) -> None:
    """Print the most recent *count* memories from the mem0 Qdrant vector store for *user_id*.

    Results are sorted by ``created_at`` descending (newest first). *fetch_limit* is how many
    points ``get_all`` retrieves before sorting; increase if you have more vectors than this.
    """
    try:
        raw = memory.get_all(user_id=user_id, limit=fetch_limit)
    except Exception as exc:
        logger.exception("Failed to list vector memories: %s", exc)
        print(f"Failed to list vector memories: {exc}", file=sys.stderr, flush=True)
        return

    memories = raw.get("results", []) if isinstance(raw, dict) else (raw or [])
    if not memories:
        print(f"\n--- Vector store: no memories for user_id={user_id!r} ---\n", flush=True)
        return

    sorted_mem = sorted(
        memories,
        key=lambda m: (m.get("created_at") or "", m.get("id") or ""),
        reverse=True,
    )
    last_n = sorted_mem[:count]

    width = 72
    print("\n" + "=" * width, flush=True)
    print(
        f"Vector store (Qdrant): newest {len(last_n)} records "
        f"(from {len(memories)} fetched, user_id={user_id!r})",
        flush=True,
    )
    print("=" * width, flush=True)
    for i, m in enumerate(last_n, 1):
        mid = m.get("id", "?")
        created = m.get("created_at", "?")
        text = m.get("memory", "") or ""
        if len(text) > 500:
            text = text[:500] + "..."
        print(f"\n--- [{i}] id={mid}", flush=True)
        print(f"    created_at={created}", flush=True)
        print(f"    memory: {text}", flush=True)
    print("=" * width + "\n", flush=True)


def main() -> None:
    global mem0_client, _available_states, _city_to_state, _city_lower_to_state

    run_dir = create_autoglm_run_dir(prefix="main_mem0")
    os.environ["AUTOGLM_RUN_OUTPUT_DIR"] = str(run_dir)
    # stdout + logging.INFO 共用 stdout_log.stdout_to_log_file 打开的单一文件句柄
    with stdout_to_log_file(run_dir, prefix="main_mem0"):
        runtime_stats.reset()
        _install_openai_usage_patch()
        _quiet_third_party_loggers()
        try:
            _build_meta_indices()
            _require_openai_config()
            _require_neo4j_config()
            _init_mem0_client()

            pattern = DefaultPattern(
                initial_agent=pref_agent,
                agents=[
                    pref_agent,
                    planner_agent,
                    poi_agent,
                    critic_agent,
                    route_timing_agent,
                ],
                user_agent=customer,
                context_variables=trip_context,
                group_after_work=TerminateTarget(),
            )

            chat_result, context_variables, last_agent = initiate_group_chat(
                pattern=pattern,
                messages="I want to visit California for 4 days. Can you help me plan my trip?",
                max_rounds=20,  #最大轮数，默认20轮
            )

            final_nl = _last_planner_agent_reply_text(chat_result)
            if final_nl:
                print(final_nl, flush=True)
            else:
                print(
                    "No planner_agent natural-language reply found in chat_history.",
                    flush=True,
                )

            if mem0_client is not None:
                print_last_vector_store_memories(mem0_client, USER_ID, count=20)
        finally:
            print(runtime_stats.format_summary())


if __name__ == "__main__":
    xhs_client = _get_xhs_mem0_client()
    # _prepare_mem0_runtime(xhs_client)
    # main()
    run_dir = create_autoglm_run_dir(prefix="main_mem0_xhs")
    os.environ["AUTOGLM_RUN_OUTPUT_DIR"] = str(run_dir)
    with stdout_to_log_file(run_dir, prefix="main_mem0_xhs"):
        runtime_stats.reset()
        _install_openai_usage_patch()
        _quiet_third_party_loggers()
        #todo：llm拆解
        #1. 解析任务请求
        task_request = parse_xhs_task_request(USER_QUERY)   
        print(
            "XHS task parsed: "
            f"destination={task_request['destination']}, "
            f"trip_days={task_request['trip_days']}, "
            f"user_query={task_request['user_query']}",
            flush=True,
        )
        #2. 获取或创建xhs查询
        exact_cache_path = resolve_xhs_query_cache_path(task_request["user_query"])
        if USE_EXISTING_GROUNDING_FILES and not exact_cache_path.is_file():
            fallback_query_result = find_cached_xhs_query_result_for_destination(task_request["destination"])
            if fallback_query_result is not None:
                print(
                    "[小红书查询] 半降级模式启用：当前 query 无精确缓存，复用同目的地已有缓存结果。"
                    f" meta={fallback_query_result.get('meta_path', 'n/a')}",
                    flush=True,
                )
                query_result = fallback_query_result
            else:
                query_result = get_or_create_xhs_query_result_json(
                    user_query=task_request["user_query"],
                    destination=task_request["destination"],
                    trip_days=task_request["trip_days"],
                )
        else:
            query_result = get_or_create_xhs_query_result_json(
                user_query=task_request["user_query"],
                destination=task_request["destination"],
                trip_days=task_request["trip_days"],
            )
        ingest_result = {
            "json_path": str(query_result["json_path"]),
        }
        #3. 将查询结果添加到向量数据库 / 图数据库（命中缓存时可跳过重复入库）
        if not (SKIP_INGEST_ON_CACHE_HIT and query_result["cache_hit"]):
            add_autoglm_result_to_mem0(
                json_path=query_result["json_path"],
                mem0_client=xhs_client,
                destination=task_request["destination"],
                metadata={
                    "query_cache_key": query_result["cache_key"],
                    "query_cache_hit": query_result["cache_hit"],
                    "user_query": task_request["user_query"],
                },
                vector_infer=False,
                write_structured_graph=True,
            )
        try:
            #4. 约束式行程生成
            run_xhs_full_itinerary_flow(
                user_query=task_request["user_query"],
                destination=task_request["destination"],
                trip_days=task_request["trip_days"],
                autoglm_ingest_result=ingest_result,
                mem0_client=xhs_client,
            )
        finally:
            stats = runtime_stats.snapshot()
            successful_llm_calls = stats["llm_call_count"] - stats["llm_failure_count"]
            print(
                f"[运行总结] 模型成功调用次数：{successful_llm_calls} 次。",
                flush=True,
            )
            print(runtime_stats.format_summary())

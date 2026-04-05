"""
Travel Planner — mem0 + Neo4j edition.

Replaces FalkorDB with mem0 vector memory (user preferences) and a Neo4j
knowledge graph (Attraction → City → State, Attraction → Category, NEAR).

Usage:
    1. Run preload_memories.py once to ingest user review preferences.
    2. python main_mem0.py

Environment (optional):
    MEM0_SKIP_PRELOAD=1 — 跳过把 review 写入 mem0（需已预载过或仅用无偏好流程）。
    若卡在 preload：mem0.add 会并行写 Qdrant + Neo4j，Neo4j 不可达时会长时间阻塞。
"""

import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from dotenv import load_dotenv

load_dotenv()

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

from itinerary_models import Itinerary, update_itinerary_with_travel_times
from graph_builder import (
    build_available_states,
    build_city_to_state_index,
    prepare_state_graph_data,
)
from preload_memories import preload_reviews

from stdout_log import stdout_to_log_file

logger = logging.getLogger(__name__)

# Set in main() before chat / graph tools run
mem0_client = None
_available_states: Dict[str, str] = {}
_city_to_state: Dict[str, str] = {}
_city_lower_to_state: Dict[str, str] = {}

# =====================================================================
# 1. Configuration
# =====================================================================

api_key = os.environ.get("OPENAI_API_KEY")

llm_config = LLMConfig(
    model="gpt-4o-mini",
    api_key=api_key,
    timeout=120,
    max_tokens=5000,
)

USER_ID = os.environ.get("MEM0_USER_ID", "bryce_caster")

META_DIR = Path(
    os.environ.get(
        "META_DIR",
        "/data/lrh/InteRecAgent/resources/google/googlelocal_data/business",
    )
)

# =====================================================================
# 2. mem0 config (client created in main())
# =====================================================================

mem0_config = {
    "llm": {
        "provider": "openai",
        "config": {
            "model": "gpt-4o-mini",
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
            "url": os.environ.get("NEO4J_URL", "bolt://10.176.25.117:8687"),
            "username": os.environ.get("NEO4J_USERNAME"),
            "password": os.environ.get("NEO4J_PASSWORD"),
            "database": os.environ.get("NEO4J_DATABASE"),
        },
    },
}

REVIEWS_PATH = Path(
    os.environ.get(
        "REVIEWS_PATH",
        "/data/lrh/InteRecAgent/resources/google/output/User/"
        "1_user_110127197526819446448/reviews.jsonl",
    )
)


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

# user_preferences is intentionally empty here; pref_agent will populate
# it from the vector store when the first user message arrives.
trip_context = ContextVariables(
    data={
        "itinerary_confirmed": False,
        "structured_itinerary": None,
        "user_preferences": "",
        "destination_state": None,
        "destination_city": None,
        # True after set_destination until graphrag_agent returns candidates (blocks repeat graphrag).
        "graphrag_handoff_needed": False,
        "last_user_feedback": "",
        "last_user_feedback_empty": False,
    }
)


# ---- Preference tools ----

def store_user_preference(
    preference: str, context_variables: ContextVariables
) -> ReplyResult:
    """Store a new user preference expressed during conversation."""
    mem0_client.add(preference, user_id=USER_ID)
    return ReplyResult(
        message=f"Noted preference: {preference}",
        context_variables=context_variables,
        target=StayTarget(),
    )


def set_destination(
    state: str = "",
    city: str = "",
    context_variables: Optional[ContextVariables] = None,
) -> ReplyResult:
    """Record destination before handing off to graphrag_agent.

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
    # New destination → need a fresh graphrag candidate fetch
    context_variables["graphrag_handoff_needed"] = True

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
    """Store and mark our itinerary as accepted by the customer."""
    context_variables["itinerary_confirmed"] = True
    return ReplyResult(
        message="Itinerary recorded and confirmed.",
        context_variables=context_variables,
        target=AgentNameTarget("structured_output_agent"),
    )


def create_structured_itinerary(
    structured_itinerary: str, context_variables: ContextVariables
) -> ReplyResult:
    """Format and store the structured itinerary JSON."""
    if not context_variables["itinerary_confirmed"]:
        return ReplyResult(
            message="Itinerary not confirmed, please confirm with the customer first.",
            context_variables=context_variables,
            target=AgentNameTarget("planner_agent"),
        )
    try:
        json.loads(structured_itinerary)
    except (ValueError, TypeError):
        return ReplyResult(
            message="Structured itinerary is not valid JSON, please reformat.",
            context_variables=context_variables,
            target=AgentNameTarget("structured_output_agent"),
        )
    if context_variables.get("structured_itinerary") is not None:
        return ReplyResult(
            message="Structured itinerary already stored.",
            context_variables=context_variables,
            target=AgentNameTarget("route_timing_agent"),
        )
    context_variables["structured_itinerary"] = structured_itinerary
    return ReplyResult(
        message="Structured itinerary stored.",
        context_variables=context_variables,
        target=AgentNameTarget("route_timing_agent"),
    )


def restart_trip_planning(context_variables: ContextVariables) -> ReplyResult:
    """Full restart after the customer rejects or wants to redo the plan post-itinerary."""
    context_variables["graphrag_handoff_needed"] = True
    context_variables["itinerary_confirmed"] = False
    context_variables["structured_itinerary"] = None
    return ReplyResult(
        message=(
            "Planning reset: you may adjust destination with set_destination if needed, "
            "then hand off to graphrag_agent when graphrag_handoff_needed is true."
        ),
        context_variables=context_variables,
        target=StayTarget(),
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
    "GRAPH RAG HANDOFF (important):\n"
    "- The handoff tool to graphrag_agent is only offered when graphrag_handoff_needed is true "
    "(after set_destination, or after restart_trip_planning). After graphrag returns candidates, "
    "that flag becomes false — do NOT hand off again until the destination changes (call "
    "set_destination again) or the customer needs a full replan (restart_trip_planning).\n\n"
    "WORKFLOW:\n"
    "1. Ask the customer for DESTINATION and NUMBER OF DAYS if missing.\n"
    "2. Call set_destination as soon as you have at least a state OR a city.\n"
    "3. When graphrag_handoff_needed is true, hand off to graphrag_agent once to get the Markdown "
    "candidate list for the current destination.\n"
    "4. Present a concise shortlist of POIs to the customer (from that list only).\n"
    "5. FEEDBACK after the POI shortlist:\n"
    "   - If last_user_feedback_empty is True (customer sent an empty message / skip): treat as "
    "\"no change\" and immediately proceed to build the day-by-day itinerary using ONLY POI names "
    "that appear in the graphrag Markdown already in this thread. Do NOT call graphrag again.\n"
    "   - If last_user_feedback_empty is False: interpret last_user_feedback and adjust your "
    "selected POIs using ONLY names from that same graphrag candidate list (no invented venues).\n"
    "6. Build a day-by-day itinerary using ONLY those attractions. Each event MUST have "
    "type='Attraction', 'location', 'city', and 'description'.\n"
    "7. FEEDBACK after you show the draft ITINERARY:\n"
    "   - If last_user_feedback_empty is True: treat as approval to continue — call "
    "mark_itinerary_as_complete with a short summary of the agreed plan.\n"
    "   - If last_user_feedback_empty is False: if the customer wants a full redo, call "
    "restart_trip_planning, then set_destination if the destination changes, and only hand off to "
    "graphrag when graphrag_handoff_needed is true again. For minor edits, revise the itinerary "
    "using the same candidate POI pool without re-querying the graph unless the destination changed.\n"
    "8. When the customer expresses a new preference during chat, call store_user_preference.\n\n"
    "CONTEXT (updated every turn before you speak):\n"
    "- graphrag_handoff_needed: {graphrag_handoff_needed}\n"
    "- last_user_feedback_empty: {last_user_feedback_empty}\n"
    "- last_user_feedback: {last_user_feedback}\n\n"
    "USER PREFERENCES (retrieved from travel memory):\n{user_preferences}"
)

planner_agent = ConversableAgent(
    name="planner_agent",
    system_message=PLANNER_SYSTEM_TEMPLATE,
    llm_config=llm_config,
    functions=[
        mark_itinerary_as_complete,
        store_user_preference,
        set_destination,
        restart_trip_planning,
    ],
    update_agent_state_before_reply=[
        _planner_sync_user_feedback,
        UpdateSystemMessage(PLANNER_SYSTEM_TEMPLATE),
    ],
)

graphrag_agent = ConversableAgent(
    name="graphrag_agent",
    system_message=(
        "Return candidate attractions for the requested location (city and/or state) as **Markdown**. "
        "Each **primary** POI (preference category match) is followed by **Nearby (NEAR graph)** POIs "
        "that are neighbours of that primary — use only names from this document as-is."
    ),
    llm_config=False,
)


# ---------------------------------------------------------------------------
# pref_agent: loads the ≤10 latest user memories from Qdrant, formats them as
# natural text, writes to context_variables["user_preferences"], then hands off
# to planner_agent.  No LLM call is made; custom reply only.
# ---------------------------------------------------------------------------

def _pref_agent_reply(
    recipient: ConversableAgent,
    messages=None,
    sender=None,
    config=None,
) -> tuple:
    raw = mem0_client.get_all(user_id=USER_ID, limit=100)
    memories = raw.get("results", []) if isinstance(raw, dict) else (raw or [])
    # Sort by created_at descending to get the most recent entries
    memories = sorted(memories, key=lambda m: m.get("created_at") or "", reverse=True)[:10]
    if memories:
        lines = [f"{i}. {m['memory']}" for i, m in enumerate(memories, 1)]
        prefs_text = "Here are the user's latest travel memories:\n" + "\n".join(lines)
    else:
        prefs_text = "No travel history found for this user."
    cv = getattr(recipient, "context_variables", None)
    if cv is not None:
        cv["user_preferences"] = prefs_text
    return True, prefs_text


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

structured_output_agent = ConversableAgent(
    name="structured_output_agent",
    system_message=(
        "You are a data formatting agent. Format the itinerary from the "
        "conversation into the required structured format, then YOU MUST call "
        "the create_structured_itinerary tool with the resulting JSON string."
    ),
    llm_config=LLMConfig(
        model="gpt-4o-mini",
        api_key=api_key,
        response_format=Itinerary,
        timeout=120,
        max_tokens=5000,
    ),
    functions=[create_structured_itinerary],
)

route_timing_agent = ConversableAgent(
    name="route_timing_agent",
    system_message=(
        "You are a route timing agent. YOU MUST call the "
        "update_itinerary_with_travel_times tool if you do not see the exact "
        "phrase 'Timed itinerary added to context with travel times' in this "
        "conversation. Only after this please tell the customer "
        "'Your itinerary is ready!'."
    ),
    llm_config=llm_config,
    functions=[update_itinerary_with_travel_times],
)


# =====================================================================
# 6. Custom reply for graphrag_agent (replaces FalkorGraphRagCapability)
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


def graphrag_reply(
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

    # Step 3: Get user preferences from context variables
    prefs = ""
    cv = getattr(recipient, "context_variables", None)
    if cv and hasattr(cv, "get"):
        prefs = cv.get("user_preferences", "")

    # Step 4: Match top-50 categories by embedding similarity (no threshold)
    matched_cats_scored = (
        mem0_client.match_attraction_categories(prefs, limit=50) if prefs else []
    )
    logger.info(
        "State=%s, City=%s, matched categories (top-5 of %d): %s",
        state, city, len(matched_cats_scored),
        [(c, round(s, 3)) for c, s in matched_cats_scored[:5]],
    )

    # Step 5: Query by city or statewide → returns (direct, near)
    if city:
        direct, near = mem0_client.query_candidate_attractions(city, matched_cats_scored)
    else:
        direct, near = mem0_client.query_candidate_attractions_in_state(
            state, matched_cats_scored
        )

    # Step 6: Format and return
    body = _format_attraction_list(direct, near, city, state)
    if cv is not None:
        cv["graphrag_handoff_needed"] = False
        cv["graphrag_candidates_markdown"] = body
    return True, body


# Register the custom reply (replaces FalkorGraphRagCapability.add_to_agent)
graphrag_agent.register_reply(
    [ConversableAgent, None],
    graphrag_reply,
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
            target=AgentTarget(graphrag_agent),
            condition=StringLLMCondition(
                "Need candidate attractions from the knowledge graph for the current destination. "
                "Only use when set_destination has already been called and the customer needs a fresh "
                "candidate list (graphrag_handoff_needed is true)."
            ),
            available=StringAvailableCondition("graphrag_handoff_needed"),
        ),
        OnCondition(
            target=AgentTarget(structured_output_agent),
            condition=StringLLMCondition(
                "Itinerary is confirmed by the customer"
            ),
        ),
    ]
)
planner_agent.handoffs.set_after_work(RevertToUserTarget())

pref_agent.handoffs.set_after_work(AgentTarget(planner_agent))
graphrag_agent.handoffs.set_after_work(AgentTarget(planner_agent))
structured_output_agent.handoffs.set_after_work(AgentTarget(route_timing_agent))
route_timing_agent.handoffs.set_after_work(TerminateTarget())


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

    if not api_key:
        print(
            "OPENAI_API_KEY not set. Copy .env.example to .env and add your key.",
            file=sys.stderr,
            flush=True,
        )
        raise ValueError(
            "OPENAI_API_KEY not set. Copy .env.example to .env and add your key."
        )

    log_dir = Path(__file__).resolve().parent / "logs"
    # stdout + logging.INFO 共用 stdout_log.stdout_to_log_file 打开的单一文件句柄
    with stdout_to_log_file(log_dir, prefix="main_mem0"):
        mem0_client = Memory.from_config(mem0_config)

        if os.environ.get("MEM0_SKIP_QDRANT_RESET", "").lower() not in ("1", "true", "yes"):
            logger.info("Resetting local Qdrant collection (mem0 vector_store.reset) ...")
            vs = mem0_client.vector_store
            if hasattr(vs, "reset"):
                vs.reset()
            else:
                logger.warning("Vector store has no reset(); skipping Qdrant clear.")

        if os.environ.get("MEM0_SKIP_PRELOAD", "").lower() in ("1", "true", "yes"):
            logger.info(
                "Skipping preload_reviews (MEM0_SKIP_PRELOAD=1). "
                "Ensure mem0 already has your vectors/graph if you need them."
            )
        else:
            logger.info("Preloading user review preferences into mem0 ...")
            print(
                "若长时间停在此处，多半是 mem0.add 在等待 Neo4j（bolt）或本地 embedding；"
                "可检查 NEO4J_URL 网络与防火墙。跳过预加载可设环境变量 MEM0_SKIP_PRELOAD=1。",
                file=sys.stderr,
                flush=True,
            )
            preload_reviews(mem0_client, reviews_path=REVIEWS_PATH, user_id=USER_ID)

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

        pattern = DefaultPattern(
            initial_agent=pref_agent,
            agents=[
                pref_agent,
                planner_agent,
                graphrag_agent,
                structured_output_agent,
                route_timing_agent,
            ],
            user_agent=customer,
            context_variables=trip_context,
            group_after_work=TerminateTarget(),
        )

        chat_result, context_variables, last_agent = initiate_group_chat(
            pattern=pattern,
            messages="I want to visit California for 4 days. Can you help me plan my trip?",
            max_rounds=100,
        )

        if "timed_itinerary" in context_variables:
            print_itinerary(context_variables["timed_itinerary"])
        else:
            print("No itinerary available to print.")

        print_last_vector_store_memories(mem0_client, USER_ID, count=20)


if __name__ == "__main__":
    main()

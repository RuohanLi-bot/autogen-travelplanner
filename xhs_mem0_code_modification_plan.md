# 小红书结构化约束图谱代码修改生成方案

## 1. 改造目标

在保留当前 `main_mem0.py` 中 AutoGLM 获取小红书帖子、mem0 写入向量库的基础上，新增一条结构化图谱写入链路：

```text
AutoGLM JSON
  -> 清洗帖子 result
  -> 写入 Qdrant 原文向量记忆
  -> 抽取结构化旅游事实
  -> 确定性归一化和校验
  -> Neo4j 写入 Post / Place / RouteVariant / Constraint / Requirement / Risk / Mitigation / Evidence
  -> RouteVariant 社区发现生成 PlayMode
  -> 用户自然语言解析成 TravelerProfile
  -> 基于帖子证据生成 FitAssessment
  -> 查询、排序并解释玩法匹配结果
```

本方案的约束：

1. 不做模型训练、微调或调参。
2. 不再依赖 mem0 通用 `MemoryGraph.add()` 自由生成实体关系来构建小红书旅游图谱。
3. LLM 只负责“证据支持的结构化抽取”，不能直接决定 Neo4j label、relationship type 或最终是否推荐。
4. Neo4j 主图使用白名单 schema，所有可计算字段必须经过确定性归一化。

## 2. 现有接入点

当前入口在 `main_mem0.py`：

```python
if __name__ == "__main__":
    _code = run_autoglm_bash_script()
    if _code != 0:
        raise SystemExit(_code)
    add_autoglm_result_to_mem0()
```

当前 `add_autoglm_result_to_mem0()` 会：

1. 解析 AutoGLM JSON。
2. 固定 `ingest_run_id = "xhs"`。
3. 创建 `Memory.from_config(mem0_config)`。
4. 设置 `client.enable_graph = True`。
5. 对每条 `result` 调用 `client.add(..., user_id=user_id, run_id="xhs")`。

需要改成：

1. 向量库仍保存 AutoGLM 原始 result，但使用 graph-disabled client，避免触发 mem0 通用图抽取。
2. 结构化图谱由新模块 `xhs_travel_graph` 写入 Neo4j。
3. 社区发现由新模块读取结构化 `RouteVariant` 后写回 `PlayMode`。

## 3. 新增文件总览

建议新增一个独立包：

```text
xhs_travel_graph/
  __init__.py
  models.py
  post_parser.py
  normalizer.py
  extractor.py
  graph_repository.py
  graph_writer.py
  cluster.py
  profile_parser.py
  fit_evaluator.py
  matcher.py
  pipeline.py
```

建议新增测试：

```text
tests/
  test_xhs_post_parser.py
  test_xhs_normalizer.py
  test_xhs_graph_writer.py
  test_xhs_cluster.py
  test_xhs_matcher.py
```

由于当前 `.gitignore` 是白名单式忽略，若这些文件需要纳入 Git，需要追加：

```gitignore
!/xhs_travel_graph/
!/xhs_travel_graph/__init__.py
!/xhs_travel_graph/models.py
!/xhs_travel_graph/post_parser.py
!/xhs_travel_graph/normalizer.py
!/xhs_travel_graph/extractor.py
!/xhs_travel_graph/graph_repository.py
!/xhs_travel_graph/graph_writer.py
!/xhs_travel_graph/cluster.py
!/xhs_travel_graph/profile_parser.py
!/xhs_travel_graph/fit_evaluator.py
!/xhs_travel_graph/matcher.py
!/xhs_travel_graph/pipeline.py
!/tests/
!/tests/test_xhs_post_parser.py
!/tests/test_xhs_normalizer.py
!/tests/test_xhs_graph_writer.py
!/tests/test_xhs_cluster.py
!/tests/test_xhs_matcher.py
!/xhs_mem0_code_modification_plan.md
```

## 4. 依赖调整

当前环境已验证 `networkx 3.2.1` 可用，但项目 `requirements.txt` 和 `pyproject.toml` 没有声明。建议显式增加：

```text
networkx>=3.2
```

如使用直接 Neo4j driver，也需要：

```text
neo4j>=5.23
```

但更推荐先复用 mem0 已创建的 Neo4j query runner，避免引入新的连接配置路径。直接 Neo4j driver 可作为后续独立脚本方案。

## 5. 数据模型设计

新增 `xhs_travel_graph/models.py`。

### 5.1 帖子模型

```python
from __future__ import annotations

from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class XHSPostEvidence(BaseModel):
    post_id: str
    run_id: str = "xhs"
    source_file: str
    result_index: int
    result_count: int
    task: str = ""
    query: str = ""
    title: str = ""
    author: str = ""
    body: str
    raw_result: str
```

### 5.2 旅游事实模型

```python
class ConstraintFact(BaseModel):
    metric: Literal[
        "stairs",
        "duration_min",
        "duration_max_min",
        "extra_cost_cny",
        "physical_load_rank",
        "transport_mode",
        "crowd_level",
        "walk_reduction",
        "style",
    ]
    value_num: Optional[float] = None
    value_text: str = ""
    unit: str = ""
    bound: Literal["exact", "min", "max", "range", "unknown"] = "unknown"
    polarity: Literal["positive", "negative", "neutral"] = "neutral"
    evidence_span: str


class RequirementFact(BaseModel):
    requirement_type: str
    demand: str
    magnitude: Optional[float] = None
    unit: str = ""
    evidence_span: str


class RiskFact(BaseModel):
    risk_type: str
    severity: Literal["low", "medium", "high", "unknown"] = "unknown"
    reason: str = ""
    evidence_span: str


class MitigationFact(BaseModel):
    mitigation_type: str
    method: str = ""
    extra_cost_cny: Optional[float] = None
    status: Literal["available", "unavailable", "unknown"] = "unknown"
    evidence_span: str


class RouteSegmentFact(BaseModel):
    order: int
    from_place: str = ""
    to_place: str = ""
    place_names: List[str] = Field(default_factory=list)
    transport_mode: str = "unknown"
    duration_min: Optional[int] = None
    duration_max_min: Optional[int] = None
    stairs: Optional[int] = None
    extra_cost_cny: Optional[float] = None
    physical_load_rank: Optional[int] = None
    evidence_span: str = ""


class RouteAlternativeFact(BaseModel):
    option_name: str
    constraints: List[ConstraintFact] = Field(default_factory=list)
    evidence_span: str


class RouteVariantFact(BaseModel):
    route_variant_id: str
    post_id: str
    run_id: str = "xhs"
    name: str
    destination: str = ""
    places: List[str] = Field(default_factory=list)
    segments: List[RouteSegmentFact] = Field(default_factory=list)
    alternatives: List[RouteAlternativeFact] = Field(default_factory=list)
    constraints: List[ConstraintFact] = Field(default_factory=list)
    requirements: List[RequirementFact] = Field(default_factory=list)
    risks: List[RiskFact] = Field(default_factory=list)
    mitigations: List[MitigationFact] = Field(default_factory=list)
    style_tags: List[str] = Field(default_factory=list)
    evidence_span: str
```

`ConstraintFact` 只保留数值和可直接解析的事实，例如 `duration_min`、`extra_cost_cny`、`stairs`；真正的用户适配不要靠 `ConstraintFact` 直接判断，而要使用 `Requirement/Risk/Mitigation` 与用户画像做证据评估。

`physical_load_rank` 可以作为兼容旧逻辑和排序展示的粗粒度摘要字段保留，但不能单独决定“适合/不适合”。新方案的主判断依据是 `Requirement/Risk/Mitigation + Evidence + TravelerProfile -> FitAssessment`。

### 5.3 用户画像和适配评估模型

```python
class TravelerProfile(BaseModel):
    seniors_ages: List[int] = Field(default_factory=list)
    children_ages: List[int] = Field(default_factory=list)
    mobility_notes: List[str] = Field(default_factory=list)
    swimming_ability: Literal["unknown", "cannot_swim", "basic", "good"] = "unknown"
    guardian_available: Literal["unknown", "yes", "no"] = "unknown"
    budget_level: Literal["unknown", "low", "medium", "high"] = "unknown"
    pace: Literal["unknown", "relaxed", "normal", "intensive"] = "unknown"
    avoid_styles: List[str] = Field(default_factory=list)


class FitAssessment(BaseModel):
    assessment_id: str
    profile_hash: str
    route_variant_id: str
    decision: Literal["pass", "conditional", "fail", "unknown"]
    hard_fail: bool = False
    reasons: List[str] = Field(default_factory=list)
    required_actions: List[str] = Field(default_factory=list)
    missing_evidence: List[str] = Field(default_factory=list)
    evidence_used: List[str] = Field(default_factory=list)
```

用户画像模型只描述“谁在旅行”和“偏好是什么”。不要在这里预先把 50/60/70 岁映射成大量固定阈值；适配判断应在 `fit_evaluator.py` 中结合路线证据完成。

## 6. AutoGLM 帖子解析

新增 `xhs_travel_graph/post_parser.py`。

### 6.1 目标

把当前 AutoGLM JSON 的每个 `result` 拆成 `XHSPostEvidence`，去掉任务日志和平台元信息。不要把“任务已完成”“我已经成功搜索”“评论数”等文本送入结构化抽取。

### 6.2 核心函数

```python
def load_autoglm_posts(json_path: Path, run_id: str = "xhs") -> List[XHSPostEvidence]:
    ...
```

```python
def parse_autoglm_result_item(
    *,
    item: dict,
    source_file: Path,
    result_index: int,
    result_count: int,
    run_id: str = "xhs",
) -> XHSPostEvidence:
    ...
```

```python
def clean_autoglm_result(result: str) -> tuple[str, dict]:
    ...
```

### 6.3 清洗规则

优先使用正则和段落标题，不依赖 LLM：

```python
TITLE_PATTERNS = [
    r"\\*\\*标题[:：]\\*\\*\\s*(.+)",
    r"\\*\\*笔记标题\\*\\*[:：]\\s*(.+)",
]

AUTHOR_PATTERNS = [
    r"\\*\\*作者[:：]\\*\\*\\s*(.+)",
    r"\\*\\*笔记作者\\*\\*[:：]\\s*(.+)",
]

NOISE_PREFIXES = [
    "任务已完成",
    "我已经成功搜索",
    "我成功搜索了",
    "笔记内容已完整展示",
]
```

正文切片规则：

1. 若存在 `**正文内容：**`，取其后内容。
2. 若存在 `**笔记内容概要**`，取其后内容。
3. 截断 `**评论数：**` 之后的内容。
4. 删除“第 N 条帖子完整正文内容如下”这类过渡句。

`post_id` 生成：

```python
post_id = sha1(f"{source_file}:{result_index}:{title}:{author}".encode("utf-8")).hexdigest()[:16]
```

## 7. 归一化模块

新增 `xhs_travel_graph/normalizer.py`。

### 7.1 核心职责

1. 同义词归一化。
2. 数值归一化。
3. 体力强度归一化。
4. 路线节点 canonical name 归一化。
5. 自环、空实体、平台噪声过滤。

### 7.2 关键函数

```python
def normalize_place_name(name: str) -> str:
    ...
```

```python
def normalize_transport_mode(text: str) -> str:
    ...
```

```python
def parse_duration_to_minutes(text: str) -> tuple[Optional[int], Optional[int]]:
    ...
```

```python
def parse_money_cny(text: str) -> Optional[float]:
    ...
```

```python
def parse_stairs_count(text: str) -> Optional[int]:
    ...
```

```python
def infer_physical_load_rank(text: str) -> Optional[int]:
    ...
```

```python
def validate_route_variant(fact: RouteVariantFact) -> RouteVariantFact:
    ...
```

### 7.3 确定性归一化，不做最终适配判断

```python
TRANSPORT_SYNONYMS = {
    "索道": "cable_car",
    "缆车": "cable_car",
    "快线索道": "cable_car",
    "百龙天梯": "elevator",
    "百龙电梯": "elevator",
    "穿山扶梯": "escalator",
    "扶梯": "escalator",
    "环保车": "shuttle_bus",
    "景区环保车": "shuttle_bus",
    "步行": "walking",
    "爬台阶": "walking_stairs",
}
```

```python
RISK_HINTS = {
    "不累": {"risk_type": "fatigue", "severity": "low"},
    "轻松": {"risk_type": "fatigue", "severity": "low"},
    "地势平缓": {"risk_type": "mobility", "severity": "low"},
    "999级台阶": {"risk_type": "fatigue", "severity": "high", "requirement": "climb_stairs"},
    "暴走": {"risk_type": "fatigue", "severity": "high"},
    "特种兵": {"risk_type": "fatigue", "severity": "high", "style": "special_forces_checkin"},
}
```

注意：这些不是最终适配规则，只是把文本提示转换成 `RiskFact/RequirementFact` 的低成本兜底。最终是否适合某个用户画像，由 `fit_evaluator.py` 基于证据和画像输出 `FitAssessment`。例如“海边冲浪”不需要在规则表中枚举为儿童禁忌，而是抽成 `water_activity` 和 `water_safety` 风险；如果帖子缺少教练、救生衣、浅水区、年龄限制等证据，则评估器返回 `unknown` 或 `fail`。

## 8. 结构化抽取

新增 `xhs_travel_graph/extractor.py`。

### 8.1 复用现有 LLM 客户端

可直接复用 `poi_research.llm_client.OpenAILLMClient`：

```python
from poi_research.llm_client import OpenAILLMClient
```

### 8.2 核心类

```python
class XHSTravelFactExtractor:
    def __init__(self, llm_client: OpenAILLMClient):
        self.llm_client = llm_client

    def extract(self, post: XHSPostEvidence) -> List[RouteVariantFact]:
        ...
```

### 8.3 抽取输出格式

LLM 只返回 JSON，且只允许下列字段：

```json
{
  "route_variants": [
    {
      "name": "string",
      "destination": "string",
      "places": ["string"],
      "segments": [
        {
          "order": 1,
          "from_place": "string",
          "to_place": "string",
          "place_names": ["string"],
          "transport_mode": "string",
          "duration_text": "string",
          "stairs_text": "string",
          "cost_text": "string",
          "physical_load_text": "string",
          "evidence_span": "string"
        }
      ],
      "alternatives": [
        {
          "option_name": "string",
          "evidence_span": "string"
        }
      ],
      "constraints": [
        {
          "metric": "duration|stairs|extra_cost|physical_load|transport_mode|style",
          "value_text": "string",
          "evidence_span": "string"
        }
      ],
      "requirements": [
        {
          "requirement_type": "mobility|endurance|water_activity|height_exposure|time|cost|accessibility|other",
          "demand": "string",
          "magnitude": 0,
          "unit": "string",
          "evidence_span": "string"
        }
      ],
      "risks": [
        {
          "risk_type": "fatigue|water_safety|height_exposure|crowd|weather|cost|time|other",
          "severity": "low|medium|high|unknown",
          "reason": "string",
          "evidence_span": "string"
        }
      ],
      "mitigations": [
        {
          "mitigation_type": "transport_substitution|coach|safety_equipment|recommended_subset|rest|avoidance|other",
          "method": "string",
          "extra_cost_cny": 0,
          "status": "available|unavailable|unknown",
          "evidence_span": "string"
        }
      ],
      "style_tags": ["string"],
      "evidence_span": "string"
    }
  ]
}
```

LLM 输出后必须调用 `normalizer` 二次处理：

```python
raw = self.llm_client.generate_json(...)
facts = parse_route_variants(raw, post)
facts = [normalize_route_variant(fact) for fact in facts]
facts = [validate_route_variant(fact) for fact in facts if fact.evidence_span]
```

### 8.4 特殊处理“替代方案”

如果 evidence 中出现“32 元扶梯或 999 台阶”，应拆成 alternative：

```python
RouteAlternativeFact(
    option_name="escalator",
    constraints=[
        ConstraintFact(metric="extra_cost_cny", value_num=32, unit="CNY", ...),
        ConstraintFact(metric="stairs", value_text="avoided", ...),
    ],
)

RouteAlternativeFact(
    option_name="stairs",
    constraints=[
        ConstraintFact(metric="extra_cost_cny", value_num=0, unit="CNY", ...),
        ConstraintFact(metric="stairs", value_num=999, unit="steps", ...),
        ConstraintFact(metric="physical_load_rank", value_num=4, ...),
    ],
)
```

这一步让后续匹配可以回答“50/60/70 岁老人是否适合”，而不是只看到一句自然语言。

更通用的目标不是为每种活动都写规则，而是让抽取结果表达“活动提出了什么要求、有什么风险、有什么缓解措施”。例如：

```yaml
route_or_activity: 海边冲浪
requirements:
  - requirement_type: water_activity
    demand: balance_and_swimming_or_supervision
risks:
  - risk_type: water_safety
    severity: unknown
mitigations:
  - mitigation_type: coach
    status: unknown
  - mitigation_type: safety_equipment
    status: unknown
```

后续 `fit_evaluator.py` 会基于这些证据判断是否适合 5 岁小孩；如果没有儿童课程、教练、救生衣、浅水区等证据，则返回 `unknown` 或 `fail`，而不是靠人工枚举 `surfing + age <= 5`。

## 9. Neo4j 访问层

新增 `xhs_travel_graph/graph_repository.py`。

### 9.1 QueryRunner 抽象

```python
from typing import Any, Dict, List, Protocol


class QueryRunner(Protocol):
    def query(self, cypher: str, params: Dict[str, Any] | None = None) -> List[Dict[str, Any]]:
        ...
```

### 9.2 mem0 Neo4j runner

```python
class Mem0Neo4jQueryRunner:
    def __init__(self, mem0_client):
        if not getattr(mem0_client, "enable_graph", False):
            raise ValueError("mem0_client.enable_graph must be true for Neo4j writes")
        self._graph = mem0_client.graph.graph

    def query(self, cypher: str, params: dict | None = None) -> list[dict]:
        return self._graph.query(cypher, params=params or {})
```

这样生产代码复用当前 mem0 Neo4j 连接，不需要额外读取 `.env`。

## 10. 图谱写入

新增 `xhs_travel_graph/graph_writer.py`。

### 10.1 核心类

```python
class XHSTravelGraphWriter:
    def __init__(self, query_runner: QueryRunner):
        self.query_runner = query_runner

    def ensure_schema(self) -> None:
        ...

    def write_post(self, post: XHSPostEvidence) -> None:
        ...

    def write_route_variant(self, post: XHSPostEvidence, fact: RouteVariantFact) -> None:
        ...

    def write_many(self, posts: list[XHSPostEvidence], facts_by_post: dict[str, list[RouteVariantFact]]) -> None:
        ...
```

### 10.2 schema 初始化

```cypher
CREATE CONSTRAINT post_id IF NOT EXISTS
FOR (p:Post) REQUIRE p.id IS UNIQUE;
```

```cypher
CREATE CONSTRAINT route_variant_id IF NOT EXISTS
FOR (rv:RouteVariant) REQUIRE rv.id IS UNIQUE;
```

```cypher
CREATE CONSTRAINT play_mode_id IF NOT EXISTS
FOR (pm:PlayMode) REQUIRE pm.id IS UNIQUE;
```

```cypher
CREATE INDEX place_lookup IF NOT EXISTS
FOR (p:Place) ON (p.name, p.run_id);
```

```cypher
CREATE INDEX constraint_lookup IF NOT EXISTS
FOR (c:Constraint) ON (c.metric, c.value_num, c.unit, c.run_id);
```

### 10.3 写入 Post

```cypher
MERGE (post:Post {id: $post_id})
SET post.run_id = $run_id,
    post.source_file = $source_file,
    post.result_index = $result_index,
    post.result_count = $result_count,
    post.task = $task,
    post.query = $query,
    post.title = $title,
    post.author = $author,
    post.body = $body;
```

### 10.4 写入 RouteVariant 和 Place

```cypher
MERGE (rv:RouteVariant {id: $route_variant_id})
SET rv.run_id = $run_id,
    rv.post_id = $post_id,
    rv.name = $name,
    rv.destination = $destination,
    rv.places = $places,
    rv.style_tags = $style_tags,
    rv.physical_load_rank = $physical_load_rank,
    rv.duration_min = $duration_min,
    rv.duration_max_min = $duration_max_min,
    rv.cost_min_cny = $cost_min_cny,
    rv.cost_max_cny = $cost_max_cny,
    rv.evidence_span = $evidence_span
WITH rv
MATCH (post:Post {id: $post_id})
MERGE (post)-[:DESCRIBES]->(rv);
```

对每个 place：

```cypher
MERGE (place:Place {name: $place_name, run_id: $run_id})
WITH place
MATCH (rv:RouteVariant {id: $route_variant_id})
MERGE (rv)-[:IN_PLACE]->(place);
```

### 10.5 写入 Constraint / Requirement / Risk / Mitigation 和 Evidence

```cypher
MERGE (ev:Evidence {id: $evidence_id})
SET ev.run_id = $run_id,
    ev.post_id = $post_id,
    ev.text = $evidence_text,
    ev.source_file = $source_file,
    ev.result_index = $result_index;
```

```cypher
MERGE (c:Constraint {id: $constraint_id})
SET c.run_id = $run_id,
    c.metric = $metric,
    c.value_num = $value_num,
    c.value_text = $value_text,
    c.unit = $unit,
    c.bound = $bound,
    c.polarity = $polarity
WITH c
MATCH (rv:RouteVariant {id: $route_variant_id})
MERGE (rv)-[:HAS_CONSTRAINT]->(c)
WITH c
MATCH (ev:Evidence {id: $evidence_id})
MERGE (c)-[:SUPPORTED_BY]->(ev);
```

对 `Requirement/Risk/Mitigation` 使用同样的 evidence-first 写入方式：

```cypher
MERGE (req:Requirement {id: $requirement_id})
SET req.run_id = $run_id,
    req.requirement_type = $requirement_type,
    req.demand = $demand,
    req.magnitude = $magnitude,
    req.unit = $unit
WITH req
MATCH (rv:RouteVariant {id: $route_variant_id})
MERGE (rv)-[:REQUIRES]->(req)
WITH req
MATCH (ev:Evidence {id: $evidence_id})
MERGE (req)-[:SUPPORTED_BY]->(ev);
```

```cypher
MERGE (risk:Risk {id: $risk_id})
SET risk.run_id = $run_id,
    risk.risk_type = $risk_type,
    risk.severity = $severity,
    risk.reason = $reason
WITH risk
MATCH (rv:RouteVariant {id: $route_variant_id})
MERGE (rv)-[:HAS_RISK]->(risk)
WITH risk
MATCH (ev:Evidence {id: $evidence_id})
MERGE (risk)-[:SUPPORTED_BY]->(ev);
```

```cypher
MERGE (mit:Mitigation {id: $mitigation_id})
SET mit.run_id = $run_id,
    mit.mitigation_type = $mitigation_type,
    mit.method = $method,
    mit.extra_cost_cny = $extra_cost_cny,
    mit.status = $status
WITH mit
MATCH (rv:RouteVariant {id: $route_variant_id})
MERGE (rv)-[:HAS_MITIGATION]->(mit)
WITH mit
MATCH (ev:Evidence {id: $evidence_id})
MERGE (mit)-[:SUPPORTED_BY]->(ev);
```

### 10.6 写入 RouteSegment

```cypher
MERGE (seg:RouteSegment {id: $segment_id})
SET seg.run_id = $run_id,
    seg.order = $order,
    seg.from_place = $from_place,
    seg.to_place = $to_place,
    seg.place_names = $place_names,
    seg.transport_mode = $transport_mode,
    seg.duration_min = $duration_min,
    seg.duration_max_min = $duration_max_min,
    seg.stairs = $stairs,
    seg.extra_cost_cny = $extra_cost_cny,
    seg.physical_load_rank = $physical_load_rank,
    seg.evidence_span = $evidence_span
WITH seg
MATCH (rv:RouteVariant {id: $route_variant_id})
MERGE (rv)-[:HAS_SEGMENT]->(seg);
```

## 11. 社区发现生成 PlayMode

新增 `xhs_travel_graph/cluster.py`。

### 11.1 输入图构建

从 Neo4j 读取同一目的地的 `RouteVariant`：

```cypher
MATCH (rv:RouteVariant {run_id: $run_id})
WHERE $destination = "" OR rv.destination = $destination
OPTIONAL MATCH (rv)-[:HAS_SEGMENT]->(seg:RouteSegment)
OPTIONAL MATCH (rv)-[:HAS_CONSTRAINT]->(c:Constraint)
OPTIONAL MATCH (rv)-[:REQUIRES]->(req:Requirement)
OPTIONAL MATCH (rv)-[:HAS_RISK]->(risk:Risk)
OPTIONAL MATCH (rv)-[:HAS_MITIGATION]->(mit:Mitigation)
RETURN rv.id AS id,
       rv.name AS name,
       rv.destination AS destination,
       rv.places AS places,
       rv.style_tags AS style_tags,
       rv.physical_load_rank AS physical_load_rank,
       collect(DISTINCT {
           order: seg.order,
           from_place: seg.from_place,
           to_place: seg.to_place,
           transport_mode: seg.transport_mode,
           physical_load_rank: seg.physical_load_rank
       }) AS segments,
       collect(DISTINCT {
           metric: c.metric,
           value_num: c.value_num,
           value_text: c.value_text
       }) AS constraints,
       collect(DISTINCT {
           requirement_type: req.requirement_type,
           demand: req.demand,
           magnitude: req.magnitude,
           unit: req.unit
       }) AS requirements,
       collect(DISTINCT {
           risk_type: risk.risk_type,
           severity: risk.severity
       }) AS risks,
       collect(DISTINCT {
           mitigation_type: mit.mitigation_type,
           method: mit.method,
           status: mit.status
       }) AS mitigations;
```

在 Python 侧建立无向加权图：

```python
def build_route_similarity_graph(route_rows: list[dict]) -> nx.Graph:
    graph = nx.Graph()
    for row in route_rows:
        graph.add_node(row["id"], **row)
    for a, b in combinations(route_rows, 2):
        weight, reasons = route_similarity(a, b)
        if weight > 0:
            graph.add_edge(a["id"], b["id"], weight=weight, reasons=reasons)
    return graph
```

### 11.2 相似度规则

权重不使用训练参数，只使用可解释证据计数：

```python
def route_similarity(a: dict, b: dict) -> tuple[int, list[str]]:
    reasons = []
    if shared_places(a, b):
        reasons.append("shared_place")
    if shared_route_bigrams(a, b):
        reasons.append("shared_route_bigram")
    if shared_transport_modes(a, b):
        reasons.append("shared_transport_mode")
    if same_risk_or_requirement_bucket(a, b):
        reasons.append("same_fatigue_or_mobility_risk_bucket")
    if shared_requirement_types(a, b):
        reasons.append("shared_requirement_type")
    if shared_mitigation_types(a, b):
        reasons.append("shared_mitigation_type")
    if shared_style_tags(a, b):
        reasons.append("shared_style_tag")
    return len(reasons), reasons
```

例子：

```text
Route A: 天子山索道 -> 杨家界 -> 袁家界 -> 百龙天梯
Route B: 东门 A 线 -> 天子山 -> 杨家界 -> 袁家界 -> 百龙电梯

shared_place = true
shared_route_bigram = true
shared_transport_mode = true
same_physical_load_bucket = true
weight = 4
```

### 11.3 社区发现算法

优先：

```python
communities = nx.algorithms.community.louvain_communities(
    graph,
    weight="weight",
    seed=42,
)
```

兜底：

```python
communities = nx.algorithms.community.greedy_modularity_communities(
    graph,
    weight="weight",
)
```

说明：

1. 当前 Neo4j GDS 不可用，所以应用侧 NetworkX 是默认实现。
2. `seed=42` 只用于稳定 Louvain 输出，不是业务调参。
3. 社区数量由图结构决定，不手工指定 k，因此不是 KMeans 这类需要指定簇数的方法。

### 11.4 PlayMode 汇总

对每个社区生成一个 `PlayMode`：

```python
class PlayModeSummary(BaseModel):
    play_mode_id: str
    run_id: str
    name: str
    destination: str
    route_variant_ids: list[str]
    representative_places: list[str]
    dominant_transport_modes: list[str]
    style_tags: list[str]
    physical_load_rank: int | None
    duration_min: int | None
    duration_max_min: int | None
    cost_min_cny: float | None
    cost_max_cny: float | None
    evidence_count: int
```

命名规则：

```python
name = f"{destination}-{top_places}-{dominant_transport}-{dominant_style}"
```

例如：

```text
张家界森林公园-天子山/袁家界-索道电梯-亲子轻松线
```

### 11.5 写回 PlayMode

```cypher
MERGE (pm:PlayMode {id: $play_mode_id})
SET pm.run_id = $run_id,
    pm.name = $name,
    pm.destination = $destination,
    pm.representative_places = $representative_places,
    pm.dominant_transport_modes = $dominant_transport_modes,
    pm.style_tags = $style_tags,
    pm.physical_load_rank = $physical_load_rank,
    pm.duration_min = $duration_min,
    pm.duration_max_min = $duration_max_min,
    pm.cost_min_cny = $cost_min_cny,
    pm.cost_max_cny = $cost_max_cny,
    pm.evidence_count = $evidence_count;
```

```cypher
MATCH (pm:PlayMode {id: $play_mode_id})
MATCH (rv:RouteVariant {id: $route_variant_id})
MERGE (pm)-[:CONTAINS]->(rv);
```

## 12. 用户画像解析、证据适配评估和匹配

新增 `xhs_travel_graph/profile_parser.py`、`xhs_travel_graph/fit_evaluator.py` 和 `xhs_travel_graph/matcher.py`。

### 12.1 用户画像解析

核心函数：

```python
def parse_traveler_profile(user_text: str) -> TravelerProfile:
    ...
```

先用轻量规则抽取明确字段，如年龄、预算、节奏、行动能力描述；这一步只解析画像，不做路线适配判断：

```python
AGE_PATTERN = re.compile(r"(\\d{1,3})\\s*岁")
CHILD_KEYWORDS = ["孩子", "小孩", "儿童", "带娃", "亲子"]
SENIOR_KEYWORDS = ["老人", "父母", "爸妈", "长辈", "腿脚不便"]
BUDGET_LOW_KEYWORDS = ["省钱", "经济", "便宜", "预算低"]
RELAXED_KEYWORDS = ["轻松", "休闲", "不累", "慢节奏"]
```

如果用户画像有复杂表达，例如“孩子会游泳但老人腿脚不便”，可以让 LLM 输出固定 JSON，再由 Pydantic 校验：

```python
{
  "seniors_ages": [70],
  "children_ages": [12, 5],
  "mobility_notes": [],
  "swimming_ability": "unknown",
  "guardian_available": "unknown",
  "budget_level": "low",
  "pace": "relaxed",
  "avoid_styles": []
}
```

### 12.2 证据适配评估器

核心函数：

```python
class FitEvaluator:
    def __init__(self, llm_client: OpenAILLMClient):
        self.llm_client = llm_client

    def evaluate_route_variant(
        self,
        profile: TravelerProfile,
        route_payload: dict,
    ) -> FitAssessment:
        ...
```

输入给评估器的 `route_payload` 必须只包含结构化 facts 和原文 evidence：

```yaml
route_variant_id: rv_123
name: 天门山穿山扶梯段
requirements:
  - requirement_type: mobility
    demand: climb_stairs
    magnitude: 999
    unit: steps
risks:
  - risk_type: fatigue
    severity: high
mitigations:
  - mitigation_type: transport_substitution
    method: escalator
    extra_cost_cny: 32
    status: available
evidence:
  - "穿山扶梯需买票32元或爬999台阶"
```

输出固定 JSON：

```json
{
  "decision": "conditional",
  "hard_fail": false,
  "reasons": [
    "999级台阶对70岁老人和轻松游画像风险高；选择32元扶梯替代后可保留该路段。"
  ],
  "required_actions": ["选择扶梯，不选择爬999级台阶"],
  "missing_evidence": [],
  "evidence_used": ["穿山扶梯需买票32元或爬999台阶"]
}
```

对于“海边冲浪 + 5 岁儿童”，如果 evidence 只写“海边冲浪”，评估器应输出：

```json
{
  "decision": "unknown",
  "hard_fail": true,
  "reasons": [
    "帖子缺少5岁儿童适配、安全装备、教练或浅水区证据，不能判为适合。"
  ],
  "required_actions": [],
  "missing_evidence": ["child_age_min", "coach_available", "safety_equipment", "shallow_water_area"],
  "evidence_used": ["海边冲浪"]
}
```

程序侧只保留少量不可绕过的安全底线：

```text
如果 decision = unknown 且涉及 low-age child + water_safety / high_exposure / traffic_safety 等高风险类型，则不能推荐为主行程，只能提示证据不足。
如果 decision = conditional，则必须把 required_actions 写入最终行程说明。
如果 decision = fail，则从候选主路线中过滤。
```

这避免为每个活动堆 `if/else`，同时保证 LLM 不会黑盒直接决定最终路线。

### 12.3 查询和评估 PlayMode

```cypher
MATCH (pm:PlayMode {run_id: $run_id})
WHERE $destination = "" OR pm.destination = $destination
OPTIONAL MATCH (pm)-[:CONTAINS]->(rv:RouteVariant)
OPTIONAL MATCH (rv)-[:HAS_CONSTRAINT]->(c:Constraint)
OPTIONAL MATCH (rv)-[:REQUIRES]->(req:Requirement)-[:SUPPORTED_BY]->(req_ev:Evidence)
OPTIONAL MATCH (rv)-[:HAS_RISK]->(risk:Risk)-[:SUPPORTED_BY]->(risk_ev:Evidence)
OPTIONAL MATCH (rv)-[:HAS_MITIGATION]->(mit:Mitigation)-[:SUPPORTED_BY]->(mit_ev:Evidence)
OPTIONAL MATCH (c)-[:SUPPORTED_BY]->(ev:Evidence)
RETURN pm.id AS play_mode_id,
       pm.name AS name,
       pm.destination AS destination,
       pm.physical_load_rank AS physical_load_rank,
       pm.duration_min AS duration_min,
       pm.duration_max_min AS duration_max_min,
       pm.cost_min_cny AS cost_min_cny,
       pm.cost_max_cny AS cost_max_cny,
       pm.evidence_count AS evidence_count,
       collect(DISTINCT rv.id) AS route_variant_ids,
       collect(DISTINCT {
           metric: c.metric,
           value_num: c.value_num,
           value_text: c.value_text,
           unit: c.unit,
           evidence: ev.text
       }) AS constraints,
       collect(DISTINCT {
           requirement_type: req.requirement_type,
           demand: req.demand,
           magnitude: req.magnitude,
           unit: req.unit,
           evidence: req_ev.text
       }) AS requirements,
       collect(DISTINCT {
           risk_type: risk.risk_type,
           severity: risk.severity,
           evidence: risk_ev.text
       }) AS risks,
       collect(DISTINCT {
           mitigation_type: mit.mitigation_type,
           method: mit.method,
           status: mit.status,
           extra_cost_cny: mit.extra_cost_cny,
           evidence: mit_ev.text
       }) AS mitigations;
```

Python 侧执行证据评估、最小安全底线过滤和排序：

```python
def match_play_modes(rows: list[dict], profile: TravelerProfile) -> list[MatchResult]:
    results = []
    for row in rows:
        result = evaluate_play_mode(row, profile)
        if result.assessment.decision != "fail" and not result.blocked_by_safety_floor:
            results.append(result)
    return pareto_then_lexicographic_sort(results)
```

排序键：

```python
(
    result.decision_rank,  # pass < conditional < unknown < fail
    int(result.assessment.hard_fail),
    result.missing_required_evidence_count,
    result.unresolved_risk_count,
    result.required_action_count,
    result.cost_max_cny or 999999,
    result.duration_max_min or 999999,
    -result.evidence_count,
)
```

## 13. pipeline 编排

新增 `xhs_travel_graph/pipeline.py`。

### 13.1 核心函数

```python
def ingest_autoglm_json_to_structured_xhs_graph(
    *,
    json_path: Path,
    mem0_client,
    run_id: str = "xhs",
    destination: str = "",
    write_schema: bool = True,
    cluster_play_modes: bool = True,
) -> dict:
    posts = load_autoglm_posts(json_path, run_id=run_id)
    extractor = XHSTravelFactExtractor(OpenAILLMClient())
    facts_by_post = {}
    for post in posts:
        facts_by_post[post.post_id] = extractor.extract(post)

    runner = Mem0Neo4jQueryRunner(mem0_client)
    writer = XHSTravelGraphWriter(runner)
    if write_schema:
        writer.ensure_schema()
    writer.write_many(posts, facts_by_post)

    cluster_summary = {}
    if cluster_play_modes:
        cluster_summary = XHSPlayModeClusterer(runner).cluster_and_write(
            run_id=run_id,
            destination=destination,
        )

    return {
        "posts": len(posts),
        "route_variants": sum(len(v) for v in facts_by_post.values()),
        "play_modes": cluster_summary.get("play_modes", 0),
    }
```

## 14. 修改 main_mem0.py

### 14.1 imports

新增：

```python
from xhs_travel_graph.pipeline import ingest_autoglm_json_to_structured_xhs_graph
```

### 14.2 新增 vector-only client helper

```python
def _make_mem0_client(*, enable_graph: bool) -> Memory:
    cfg = json.loads(json.dumps(mem0_config))
    if not enable_graph:
        cfg["graph_store"]["config"] = {}
    client = Memory.from_config(cfg)
    client.enable_graph = enable_graph
    return client
```

如果 `graph_store.config = {}` 触发配置校验问题，则改为：

```python
client = Memory.from_config(mem0_config)
client.enable_graph = False
```

注意第二种写法仍会初始化 `client.graph`，但不会执行 `_add_to_graph()`；如果 Neo4j 不可达，第一种更安全。

### 14.3 修改 add_autoglm_result_to_mem0 参数

建议改成：

```python
def add_autoglm_result_to_mem0(
    json_path: Optional[Path] = None,
    *,
    run_open_autoglm_first: bool = False,
    extra_argv: Optional[List[str]] = None,
    user_id: str = USER_ID,
    metadata: Optional[Dict[str, Any]] = None,
    write_vector: bool = True,
    vector_infer: bool = False,
    write_legacy_graph: bool = False,
    write_structured_graph: bool = True,
    cluster_play_modes: bool = True,
    destination: str = "",
) -> Dict[str, Any]:
```

默认：

1. `write_vector=True`：保留原始小红书 result 向量化。
2. `vector_infer=False`：直接保存原文，避免 mem0 事实抽取再改写文本。
3. `write_legacy_graph=False`：默认不使用 mem0 通用图抽取。
4. `write_structured_graph=True`：默认写结构化旅游图谱。

### 14.4 vector 写入逻辑

```python
responses = []
if write_vector:
    vector_client = _make_mem0_client(enable_graph=write_legacy_graph)
    for idx, result_text in enumerate(result_texts):
        chunk_meta = {
            **base_meta,
            "result_index": idx,
            "result_count": n,
        }
        response = vector_client.add(
            messages=[{"role": "user", "content": result_text}],
            user_id=user_id,
            run_id=ingest_run_id,
            metadata=chunk_meta,
            infer=vector_infer,
        )
        responses.append(response)
```

如果用户仍需要复现实验中的旧图，可显式设置：

```text
write_legacy_graph=True
```

### 14.5 structured graph 写入逻辑

```python
structured_result = {}
if write_structured_graph:
    graph_client = _make_mem0_client(enable_graph=True)
    structured_result = ingest_autoglm_json_to_structured_xhs_graph(
        json_path=json_path,
        mem0_client=graph_client,
        run_id=ingest_run_id,
        destination=destination,
        cluster_play_modes=cluster_play_modes,
    )

return {
    "responses": responses,
    "count": len(responses),
    "structured_graph": structured_result,
}
```

### 14.6 环境变量开关

建议增加：

```python
write_structured_graph = write_structured_graph and not _env_flag("MEM0_XHS_DISABLE_STRUCTURED_GRAPH")
write_legacy_graph = write_legacy_graph or _env_flag("MEM0_XHS_LEGACY_GRAPH")
cluster_play_modes = cluster_play_modes and not _env_flag("MEM0_XHS_SKIP_CLUSTER")
```

## 15. 验证脚本和命令

### 15.1 单元测试

```bash
pytest tests/test_xhs_post_parser.py
pytest tests/test_xhs_normalizer.py
pytest tests/test_xhs_cluster.py
pytest tests/test_xhs_matcher.py
```

### 15.2 集成 smoke test

新增一个不写 Neo4j 的 dry-run：

```python
def dry_run_autoglm_json(json_path: Path) -> dict:
    posts = load_autoglm_posts(json_path)
    extractor = XHSTravelFactExtractor(OpenAILLMClient())
    facts = [extractor.extract(post) for post in posts[:3]]
    return {"posts": len(posts), "sample_facts": facts}
```

命令：

```bash
/data/lrh/anaconda3/envs/interecagent/bin/python - <<'PY'
from pathlib import Path
from xhs_travel_graph.pipeline import dry_run_autoglm_json
print(dry_run_autoglm_json(Path("/data/lrh/Open-AutoGLM-main/output/autoglm_run_20260409_223230.json")))
PY
```

### 15.3 Neo4j 验证 Cypher

检查结构化节点：

```cypher
MATCH (n)
WHERE n.run_id = "xhs"
  AND any(label IN labels(n) WHERE label IN ["Post", "Place", "RouteVariant", "Constraint", "Evidence", "PlayMode"])
RETURN labels(n) AS labels, count(*) AS count
ORDER BY count DESC;
```

检查旧污染是否减少：

```cypher
MATCH (n {run_id: "xhs"})
WHERE n.name STARTS WITH "#" OR n.name CONTAINS "任务已完成" OR n.name CONTAINS "评论数"
RETURN n.name AS name, labels(n) AS labels
LIMIT 50;
```

检查自环：

```cypher
MATCH (n {run_id: "xhs"})-[r]->(n)
WHERE r.valid IS NULL OR r.valid = true
RETURN n.name AS name, labels(n) AS labels, type(r) AS relationship
LIMIT 50;
```

结构化主图预期不应产生自环。

检查 PlayMode：

```cypher
MATCH (pm:PlayMode {run_id: "xhs"})-[:CONTAINS]->(rv:RouteVariant)
RETURN pm.name AS play_mode,
       pm.physical_load_rank AS physical_load_rank,
       pm.cost_max_cny AS cost_max_cny,
       count(rv) AS route_variants,
       collect(rv.name)[..5] AS samples
ORDER BY route_variants DESC;
```

## 16. 分阶段实施顺序

### Phase 1：不碰现有 planner，只做结构化 ingest

1. 新增 `xhs_travel_graph/models.py`。
2. 新增 `post_parser.py`，实现 AutoGLM result 清洗。
3. 新增 `normalizer.py`，实现数值和同义词归一化。
4. 新增 `extractor.py`，复用 `OpenAILLMClient` 抽结构化 facts。
5. 新增 `graph_repository.py` 和 `graph_writer.py`，写 Neo4j 结构化 schema。
6. 新增 `pipeline.py`，串联 ingest。
7. 修改 `main_mem0.py`，让 `add_autoglm_result_to_mem0()` 支持 vector-only + structured graph。

验收：

```text
AutoGLM JSON -> Qdrant 原文记忆存在 -> Neo4j 出现 Post/RouteVariant/Constraint/Requirement/Risk/Mitigation/Evidence
```

### Phase 2：社区发现 PlayMode

1. 新增 `cluster.py`。
2. 读取 `RouteVariant` 生成应用侧 NetworkX 图。
3. Louvain 社区发现。
4. 写回 `PlayMode` 和 `CONTAINS`。

验收：

```text
同一目的地出现多个 PlayMode，且每个 PlayMode 可追溯到 RouteVariant 和 Evidence。
```

### Phase 3：用户画像解析、证据适配评估和查询

1. 新增 `profile_parser.py`。
2. 新增 `fit_evaluator.py`。
3. 新增 `matcher.py`。
4. 对“70 岁老人、12 岁和 5 岁儿童、经济、省钱、轻松游”生成 `TravelerProfile`。
5. 查询 `PlayMode`，对每个候选玩法基于 `Requirement/Risk/Mitigation/Evidence` 生成 `FitAssessment`。
6. 执行最小安全底线过滤、Pareto/字典序排序和解释输出。

验收：

```text
999 台阶方案对 70 岁老人不再靠固定 if/else 直接判断，而是生成 conditional/fail 的 FitAssessment；32 元扶梯替代方案保留并解释成本。
海边冲浪对 5 岁儿童在缺少儿童安全证据时返回 unknown/fail，并列出缺失证据。
```

### Phase 4：接入 planner

1. 在 `graphrag_reply()` 或新工具函数中调用 `matcher`。
2. 将 `PlayMode` 候选和解释写入 `context_variables["poi_research_markdown"]` 或新增字段。
3. planner 生成路线时优先使用适配后的 `PlayMode` 和 `required_actions`，而不是只使用 POI 列表。
4. `route_timing_agent` 检查交通与时间，`critic_agent` 检查 `FitAssessment` 中的 hard fail、missing evidence 和 required actions 是否被路线遵守。

验收：

```text
planner 能引用“推荐玩法簇 + 证据 + 不推荐原因”，而不是只列景点。
```

## 17. 关键边界条件

1. 如果 LLM 抽取失败：保留 Post 和 raw vector memory，但不写 RouteVariant。
2. 如果某路线缺少体力、费用或安全缓解证据：不要补造数值，写 `unknown` 并在匹配时加入 `missing_required_evidence_count` 或 `missing_evidence`。
3. 如果一个帖子只提供单点建议，比如“金鞭溪不要拿袋子喂猴”：写为 `Risk/Mitigation/Evidence` 或底层 `Constraint/Evidence`，不强行生成完整路线。
4. 如果 PlayMode 社区只有 1 个 RouteVariant：仍可写入，但 `evidence_count` 较低，排序时自然靠后。
5. 如果用户要求“特种兵打卡”：不要过滤高强度风格；如果用户同时有老人/低龄儿童，则老人儿童安全约束优先。

## 18. 预期最终调用示例

```python
result = add_autoglm_result_to_mem0(
    run_open_autoglm_first=True,
    write_vector=True,
    vector_infer=False,
    write_legacy_graph=False,
    write_structured_graph=True,
    cluster_play_modes=True,
    destination="张家界",
)
print(result["structured_graph"])
```

用户偏好查询：

```python
from xhs_travel_graph.profile_parser import parse_traveler_profile
from xhs_travel_graph.matcher import query_matching_play_modes

profile = parse_traveler_profile("70岁老人、12岁和5岁儿童、经济、省钱、轻松游")
matches = query_matching_play_modes(
    query_runner=runner,
    run_id="xhs",
    destination="张家界",
    profile=profile,
)
```

预期解释：

```text
推荐：张家界森林公园-天子山/袁家界-索道电梯-亲子轻松线
原因：
1. 当前画像包含老人和低龄儿童，评估结果为 conditional/pass。
2. 帖子证据显示该玩法可使用索道/电梯替代长距离爬升。
3. 999 级台阶方案不会作为默认选择；若选择穿山扶梯需增加 32 元。
4. 有多条小红书证据支持。
```

## 19. 不建议的实现方式

1. 不建议继续把完整 AutoGLM result 交给 mem0 `MemoryGraph.add()` 构图，因为会继续产生自由 label、自由 relation 和平台噪声节点。
2. 不建议只把 prompt 改得更复杂，因为自环、label 漂移、约束字段缺失需要写图前的确定性 schema 校验解决。
3. 不建议直接用一个加权总分排序，因为用户安全、老人儿童适配、预算之间有优先级，不是同一层级的分数。
4. 不建议用需要指定簇数的 KMeans 作为首选，因为“玩法簇数量”本身应由帖子证据图决定。
5. 不建议通过大量 `if activity == ... and age == ...` 枚举活动适配关系；应抽取开放式 `Requirement/Risk/Mitigation`，再做证据驱动的画像适配评估。

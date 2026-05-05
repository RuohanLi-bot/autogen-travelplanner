# 小红书旅行信息到可计算约束图谱的优化方案

## 1. 当前实现理解

当前 `main_mem0.py` 的小红书链路分两步：

1. `run_autoglm_bash_script()` 在 `/data/lrh/Open-AutoGLM-main` 下执行 `bash run_autoglm.sh`，由 Open-AutoGLM 搜索并读取小红书帖子，输出到 `output/autoglm_run_*.json`。
2. `add_autoglm_result_to_mem0()` 读取最新 AutoGLM JSON，把每个 `result` 文本作为一条 `mem0.add()` 写入：
   - 向量库：Qdrant，本地路径来自 `MEM0_DIR/vector_store`。
   - 图数据库：Neo4j，`run_id` 固定为 `"xhs"`，metadata 中带 `graph_label: "xhs"`、`source_file`、`result_index` 等。

mem0 的 `Memory.add()` 在 `enable_graph=True` 时会并发执行：

1. `_add_to_vector_store()`：把文本抽取成 memory 写入向量库。
2. `_add_to_graph()`：调用 `MemoryGraph.add()` 抽取实体和关系并写入 Neo4j。

当前 `MemoryGraph.add()` 的核心流程是：

1. `_retrieve_nodes_from_data(data, filters)`：用 LLM 从文本中抽实体和实体类型。
2. `_establish_nodes_relations_from_data(data, filters, entity_type_map)`：用 LLM 从实体列表和原文中抽实体关系。
3. `_search_graph_db(...)`：按 embedding 相似度找已有相关关系，用于判断是否需要将旧关系置为 `valid=false`。
4. `_add_entities(...)`：按 `user_id + run_id` merge 节点，并按 LLM 输出的关系类型创建边。关系上维护 `mentions`、`valid`、`created_at`、`updated_at`。

这意味着当前图谱不是旅游领域 schema 图，而是 mem0 通用实体关系图。实体 label 和关系 type 大量来自 LLM 输出，缺少严格的旅游约束字段。

## 2. Cypher 检查结果

我用 Neo4j 只读查询筛选了 `run_id = "xhs"` 的子图。

### 2.1 使用的检查语句

```cypher
MATCH (n {run_id: $run_id})
WITH count(n) AS nodes
MATCH (a {run_id: $run_id})-[r]->(b {run_id: $run_id})
WHERE r.valid IS NULL OR r.valid = true
RETURN nodes, count(r) AS active_relationships;
```

```cypher
MATCH (n {run_id: $run_id})
UNWIND labels(n) AS label
RETURN label, count(*) AS count
ORDER BY count DESC, label;
```

```cypher
MATCH (a {run_id: $run_id})-[r]->(b {run_id: $run_id})
WHERE r.valid IS NULL OR r.valid = true
RETURN type(r) AS relationship,
       count(*) AS count,
       sum(coalesce(r.mentions, 1)) AS mentions
ORDER BY count DESC, relationship;
```

```cypher
MATCH (n {run_id: $run_id})
OPTIONAL MATCH (n)-[r]-(m {run_id: $run_id})
WHERE r.valid IS NULL OR r.valid = true
RETURN n.name AS name,
       labels(n) AS labels,
       count(r) AS degree,
       coalesce(n.mentions, 0) AS mentions
ORDER BY degree DESC, mentions DESC, name
LIMIT 30;
```

```cypher
MATCH (a {run_id: $run_id})-[r]->(a)
WHERE r.valid IS NULL OR r.valid = true
RETURN a.name AS name,
       labels(a) AS labels,
       type(r) AS relationship,
       coalesce(r.mentions, 1) AS mentions
ORDER BY mentions DESC, name;
```

### 2.2 观察到的图结构

`xhs` 子图当前有 38 个节点、62 条有效关系。关系类型主要集中在：

| relation | count | mentions |
| --- | ---: | ---: |
| `includes_activity` | 27 | 34 |
| `has_constraint` | 20 | 25 |
| `has_tag` | 5 | 8 |
| `satisfies` | 3 | 3 |
| `interest_in` | 2 | 2 |
| `participant_in` | 2 | 2 |
| `has_author` | 1 | 1 |
| `searched_for` | 1 | 1 |
| `viewed` | 1 | 1 |

节点 label 分布中，`__User__` 有 9 个，且包含 `#徒步丈量世界`、`#武陵源`、`#轻松徒步路线`、`summer麻麻`、`春节人巨多` 等明显不是用户节点的内容。旅游实体 label 也存在不一致，例如 `scenic_areas_and_points_of_interest` 和 `scenic_areas_and_points_of_interest_(pois)` 同时出现。

高中心度节点如下：

| name | labels | degree | mentions |
| --- | --- | ---: | ---: |
| 杨家界 | `activities` | 29 | 41 |
| 张家界国家森林公园 | `scenic_areas_and_points_of_interest` | 15 | 19 |
| 两天游玩计划 | `travel_routes_and_itineraries` | 12 | 15 |
| 猴子 | `scenic_features` | 9 | 17 |
| `user_id:_bryce_caster,_run_id:_xhs` | `__User__` | 9 | 10 |

当前有效自环有 5 条：

| node | label | relationship | mentions |
| --- | --- | --- | ---: |
| 猴子 | `scenic_features` | `has_tag` | 3 |
| 两天游玩计划 | `travel_routes_and_itineraries` | `includes_activity` | 1 |
| 杨家界 | `activities` | `has_constraint` | 1 |
| 杨家界 | `activities` | `includes_activity` | 1 |
| 适合小孩 | `suitability_for_specific_groups` | `satisfies` | 1 |

Neo4j GDS 当前不可用：`CALL gds.version()` 返回 `ProcedureNotFound`。我用本地 NetworkX 3.2.1 读取有效边后运行 Louvain 社区划分，结果是 1 个连通分量被分成 5 个社区，大小为 `[18, 7, 6, 5, 2]`。这些社区主要由 `杨家界`、`张家界国家森林公园`、`两天游玩计划`、`猴子`、`金鞭溪` 等 hub 牵引，说明现有图能聚出“高频共现团”，但还不能稳定表达“玩法方案”或“约束可满足性”。

## 3. 当前方案的主要问题

1. AutoGLM 的 `result` 是“任务完成总结 + 帖子摘要”，不是结构化帖子正文。它包含“任务已完成”“我已经成功搜索”“作者”“评论数”“第 N 条笔记”等大量非旅游决策信息，直接喂给 mem0 图抽取会污染图谱。
2. 图谱 schema 由 LLM 即时生成。节点 label、关系 type 没有白名单和领域约束，导致同类概念 label 不一致、作者和 hashtag 被标成 `__User__`。
3. 当前写入逻辑没有主动过滤自环。活动和特征会形成 `杨家界 -> 杨家界`、`猴子 -> 猴子` 这类无决策价值关系。
4. 图中缺少可计算字段。用户会议中强调的步数、海拔爬升、台阶、游玩时长、交通方式、费用、适合老人/儿童等，目前大多只是自然语言节点，不能直接用于 Cypher 过滤。
5. 同一景点的不同玩法没有被建模为不同实体。例如张家界森林公园的 A 线、B 线、天子山索道、百龙天梯、金鞭溪轻松玩水等应该是不同 `RouteVariant` 或 `PlayMode`，而不是全部挂到“张家界国家森林公园/杨家界”这几个 hub 上。
6. 召回和排序缺少“约束优先级”。老人、小孩等强约束应该覆盖“想多打卡”“省钱”等软偏好，但当前图没有优先级、硬约束、软约束和违约解释。

## 4. 目标图谱形态

目标不是把 prompt 写得更长，而是把小红书网页信息转成可检索、可匹配、可解释的旅行约束图。

建议保留 mem0 的双存储，但职责分离：

1. Qdrant：保存原文和帖子摘要，用于语义召回和证据回看。
2. Neo4j：只保存 schema-bound 的结构化旅游事实、玩法、约束和证据。

建议的核心节点：

| Node | 用途 |
| --- | --- |
| `Post` | 一条小红书帖子或 AutoGLM 结果项，保存 `post_id`、`title`、`author`、`query`、`result_index`、`source_file`、`run_id`。 |
| `Place` | 景点、区域、城市、入口、观景点，如张家界国家森林公园、天门山、金鞭溪、东门。 |
| `RouteVariant` | 一种具体玩法或路线，如“森林公园东门 A 线先上后下”“天门山 B 线扶梯上山”“金鞭溪亲子轻松玩水”。 |
| `RouteSegment` | 路线中的分段，带顺序、交通方式、时长、台阶、费用等。 |
| `Constraint` | 底层可计算事实，如 `duration_min=240..300`、`stairs=999`、`extra_cost_cny=32`；不直接代表适合/不适合。 |
| `Requirement` | 路线/活动对游客提出的能力要求，如 `climb_stairs=999`、`walk_duration=240..300min`、`water_activity=true`。 |
| `Risk` | 帖子证据中可推断的风险，如 `fatigue`、`water_safety`、`height_exposure`、`crowd`、`weather_exposure`。 |
| `Mitigation` | 替代方案或缓解措施，如 `escalator + extra_cost_cny=32`、`coach_available`、`life_jacket`、`shuttle_bus`。 |
| `FitAssessment` | 针对某个用户画像和候选玩法的证据化适配结论，如 `pass`、`conditional`、`fail`、`unknown`。 |
| `PlayMode` | 由社区/聚类算法归并出的玩法簇，如“亲子轻松线”“索道省力线”“高强度打卡线”。 |
| `Evidence` | 原文片段证据，连接到 Post 和具体 Constraint/Requirement/Risk/Mitigation/Assessment。 |

建议的核心关系：

| Relationship | 含义 |
| --- | --- |
| `(Post)-[:MENTIONS]->(Place)` | 帖子提到景点。 |
| `(Post)-[:DESCRIBES]->(RouteVariant)` | 帖子描述某玩法。 |
| `(RouteVariant)-[:IN_PLACE]->(Place)` | 玩法属于某景点或区域。 |
| `(RouteVariant)-[:HAS_SEGMENT]->(RouteSegment)` | 玩法包含路线段。 |
| `(RouteSegment)-[:FROM]->(Place)`、`(RouteSegment)-[:TO]->(Place)` | 路线段起止点。 |
| `(RouteVariant)-[:HAS_CONSTRAINT]->(Constraint)` | 玩法包含底层数值事实，但不直接给出适配结论。 |
| `(RouteVariant)-[:REQUIRES]->(Requirement)` | 玩法对游客能力或条件的要求。 |
| `(RouteVariant)-[:HAS_RISK]->(Risk)` | 玩法存在的证据化风险。 |
| `(RouteVariant)-[:HAS_MITIGATION]->(Mitigation)` | 玩法可用的替代或缓解措施。 |
| `(RouteVariant)-[:ASSESSED_AS]->(FitAssessment)` | 针对特定用户画像的适配评估结果。 |
| `(PlayMode)-[:CONTAINS]->(RouteVariant)` | 聚类得到的玩法簇包含玩法。 |
| `(Constraint/Requirement/Risk/Mitigation/FitAssessment)-[:SUPPORTED_BY]->(Evidence)` | 所有事实和判断都必须由原文证据支撑。 |

关键点：不要再让 LLM 自由创造关系类型。LLM 可以参与抽取，但写入 Neo4j 前必须通过 schema 校验、字段归一化和关系白名单。

`physical_load_rank` 这类粗粒度摘要字段可以保留用于展示或粗排，但不能作为最终适配判断的唯一依据；最终判断必须来自 `Requirement/Risk/Mitigation + Evidence + TravelerProfile -> FitAssessment`。

## 5. 推荐算法方案

### 5.1 帖子清洗和证据切片

先把 AutoGLM `result` 拆成结构化 `PostEvidence`，而不是整段写入图谱：

```text
post_id = hash(source_file, result_index)
title = 从 “标题/笔记标题” 提取
author = 从 “作者/笔记作者” 提取
body = 正文内容块
comments_count = 可选
query = 从 task 或搜索语句提取
noise_blocks = ["任务已完成", "我已经成功搜索", "评论数", "笔记内容已完整展示"]
```

保留原始 `result` 到 Qdrant；图谱抽取只使用 `title + body` 和必要 metadata。这样可直接减少 `summer麻麻`、hashtag、平台日志进入旅游知识图谱的概率。

### 5.2 结构化抽取后处理

LLM 仍然可以作为信息抽取器，但输出必须是 Pydantic schema，而不是自由文本实体图。示例：

```python
class ExtractedRouteVariant(BaseModel):
    route_name: str
    destination: str
    places: list[str]
    segments: list[RouteSegmentFact]
    requirements: list[RequirementFact]
    risks: list[RiskFact]
    mitigations: list[MitigationFact]
    style_tags: list[str]
    evidence_span: str
```

抽取后必须做确定性校验：

1. `RouteVariant.route_name` 为空时用 `destination + normalized_places + transport_modes` 生成。
2. `source == target` 的关系直接拒绝，不进入 Neo4j。
3. `author`、`hashtag`、`评论数`、`任务完成语句` 只允许写到 `Post` metadata，不能成为旅游实体。
4. label 和关系必须来自白名单，未知类型进入 `Evidence` 或 `RawMention`，不直接进主图。
5. 同义词归一化，例如“索道/缆车”统一为 `cable_car`，“扶梯/穿山扶梯”统一为 `escalator`，“一日游”统一为 `one_day_trip`。

这一步不是调 prompt，而是在写图前建立强约束的数据契约。

### 5.3 用户画像解析与证据驱动适配评估

不要把用户自然语言直接映射成大量人工 `if/else` 规则。更通用的方式是把用户输入解析成“画像”，把网页文本解析成“能力要求/风险/缓解措施”，再让适配评估器基于证据判断。

```yaml
TravelerProfile:
  party:
    seniors:
      - age: 70
        mobility: unknown
    children:
      - age: 12
      - age: 5
  capability_notes:
    mobility: unknown
    swimming_ability: unknown
    guardian_available: unknown
  preferences:
    budget: low
    pace: relaxed
```

网页帖子不只抽“体力强度”，而要抽开放式的中间表示：

```yaml
RouteEvidence:
  route_or_activity: 天门山穿山扶梯段
  requirements:
    - type: mobility
      demand: climb_stairs
      magnitude: 999
      unit: steps
  risks:
    - type: fatigue
      severity: high
  mitigations:
    - type: transport_substitution
      method: escalator
      extra_cost_cny: 32
  evidence: "穿山扶梯需买票32元或爬999台阶"
```

适配评估器输出固定 JSON：

```yaml
FitAssessment:
  decision: pass | conditional | fail | unknown
  hard_fail: true | false
  reasons:
    - string
  required_actions:
    - string
  evidence_used:
    - string
  missing_evidence:
    - string
```

例如“999 台阶 + 70 岁老人 + 轻松游”不靠手写 `if age >= 70` 直接判死，而是基于证据判断：

```yaml
decision: conditional
hard_fail: false
reason: "999级台阶对70岁老人和轻松游画像风险高；若选择32元扶梯替代，则该路段可保留。"
required_action: "选择扶梯，不选择爬999级台阶"
evidence_used: ["穿山扶梯需买票32元或爬999台阶"]
```

“海边冲浪 + 5 岁儿童”也用同一套机制。如果帖子只写“海边冲浪”，没有儿童课程、教练、救生衣、浅水区等证据，则不能武断推荐：

```yaml
decision: unknown
hard_fail: true
reason: "帖子缺少5岁儿童适配、安全装备、教练或浅水区证据。"
missing_evidence: ["child_age_min", "coach_available", "safety_equipment", "shallow_water_area"]
```

人工规则只保留为最小安全底线和缺证保守策略，例如“低龄儿童水上活动缺少安全证据时不能判为适合”。路线是否适合用户主要由帖子证据、结构化事实和画像适配评估共同决定。

### 5.4 可计算指标归一化

网页文本中的模糊描述先归一化成事实、要求、风险和缓解措施，但这些字段不直接等价于“适合/不适合”：

| 原文 | 结构化字段 |
| --- | --- |
| “游玩 4-5h” | `duration_min=240`, `duration_max=300` |
| “穿山扶梯需买票 32 元或爬 999 台阶” | `Requirement(mobility, climb_stairs, 999 steps)` + `Mitigation(escalator, extra_cost_cny=32)` |
| “索道上山” | `Mitigation(cable_car, walk_reduction=true)` |
| “不累/轻松/地势平缓” | `Risk(fatigue, severity=low)` 或 `Requirement(mobility, demand=low)` |
| “暴走/特种兵/一天打卡很多景点” | `Risk(fatigue, severity=high)` + `style=special_forces_checkin` |
| “适合老人/带父母” | `EvidenceClaim(target=elderly, polarity=suitable)`，仍需结合具体要求/风险评估 |
| “娃太小建议重点玩天子山和袁家界” | `Mitigation(recommended_subset=[天子山, 袁家界])` |
| “海边冲浪” | `Requirement(water_activity)` + `Risk(water_safety, severity=unknown)`，若无儿童安全证据则保持 unknown |

指标缺失时不要补造数值。应写成 `unknown` 并保留 evidence，查询时按“缺证惩罚”处理，而不是当作满足条件。

### 5.5 用 Louvain/Leiden 发现玩法簇

针对同一目的地构建 `RouteVariant` 之间的证据图，再运行社区发现算法，把帖子中分散的路线描述合成“玩法簇”。

节点：`RouteVariant` 或更细的 `RouteEvidence`。

边：两个节点满足以下任一证据时连边，权重为满足的证据条数，而不是手工调参权重：

```text
+1 共享至少一个规范化 Place
+1 共享至少一个连续路线 bigram，例如 天子山 -> 杨家界
+1 共享主要交通方式，例如 cable_car / elevator / walking
+1 同属一个风险/能力要求桶，例如 fatigue=high 或 mobility=climb_stairs
+1 同属一个 style 桶，例如 family_relaxed / private_custom / special_forces_checkin
+1 来自同一帖子或互相引用同一证据片段
```

算法选择：

1. 优先 Leiden：社区连通性更稳定，适合后续增量维护。
2. 如果 Neo4j GDS 不可用，使用应用侧 NetworkX Louvain。当前环境已经验证 NetworkX 3.2.1 可用，且 Neo4j GDS 不可用。
3. 数据规模较小时，Louvain/Leiden 的复杂度近似随边数线性增长，适合当前小红书帖子规模；不需要模型训练。

社区输出写回图：

```cypher
MERGE (pm:PlayMode {id: $play_mode_id, run_id: $run_id})
SET pm.name = $name,
    pm.destination = $destination,
    pm.physical_load_rank = $physical_load_rank,
    pm.min_cost_cny = $min_cost_cny,
    pm.duration_max_min = $duration_max_min,
    pm.evidence_count = $evidence_count
WITH pm
MATCH (rv:RouteVariant {id: $route_variant_id, run_id: $run_id})
MERGE (pm)-[:CONTAINS]->(rv);
```

玩法簇命名不依赖模型训练。可先用簇内 PageRank/weighted degree 最高的 Place、主交通方式、强约束字段组合生成名称，例如：

```text
张家界森林公园 + 天子山/袁家界 + cable_car + family_relaxed
```

LLM 可以只用于把这个机器名润色成人类可读名称，但不能决定簇成员。

### 5.6 查询和筛选

生成 query 时不要只用 `physical_load_rank <= X` 这类单字段过滤。更稳妥的流程是先召回候选 `PlayMode/RouteVariant` 和其证据，再运行画像适配评估器，最后把评估结果写回图并排序。

候选召回 Cypher：

```cypher
MATCH (pm:PlayMode {run_id: $run_id, destination: $destination})
OPTIONAL MATCH (pm)-[:CONTAINS]->(rv:RouteVariant)
OPTIONAL MATCH (rv)-[:REQUIRES]->(req:Requirement)-[:SUPPORTED_BY]->(req_ev:Evidence)
OPTIONAL MATCH (rv)-[:HAS_RISK]->(risk:Risk)-[:SUPPORTED_BY]->(risk_ev:Evidence)
OPTIONAL MATCH (rv)-[:HAS_MITIGATION]->(mit:Mitigation)-[:SUPPORTED_BY]->(mit_ev:Evidence)
RETURN pm.id AS play_mode_id,
       pm.name AS name,
       collect(DISTINCT rv.id) AS route_variant_ids,
       collect(DISTINCT req) AS requirements,
       collect(DISTINCT risk) AS risks,
       collect(DISTINCT mit) AS mitigations,
       collect(DISTINCT req_ev.text) + collect(DISTINCT risk_ev.text) + collect(DISTINCT mit_ev.text) AS evidence_texts;
```

评估完成后写回：

```cypher
MERGE (fa:FitAssessment {id: $assessment_id})
SET fa.run_id = $run_id,
    fa.profile_hash = $profile_hash,
    fa.decision = $decision,
    fa.hard_fail = $hard_fail,
    fa.reasons = $reasons,
    fa.required_actions = $required_actions,
    fa.missing_evidence = $missing_evidence
WITH fa
MATCH (rv:RouteVariant {id: $route_variant_id})
MERGE (rv)-[:ASSESSED_AS]->(fa);
```

如果用户有老人和低龄儿童，则 “特种兵打卡” 类玩法通常会被评估为 `fail` 或 `conditional`；但其中某个低风险片段仍可被保留，只要 evidence 能支持它对当前画像可行。这对应“舍弃、降权，还是只提取子信息”的会议要求。

最终旅行路线由 planner 生成，而不是社区发现直接生成。推荐链路是：

```text
PlayMode/RouteVariant 候选
-> FitAssessment 过滤和排序
-> planner_agent 组合成 day-by-day itinerary
-> route_timing_agent 检查交通和时间
-> critic_agent 检查硬约束、证据缺口和不可执行点
-> final itinerary
```

### 5.7 排序策略：Pareto + 字典序优先级

不建议直接使用一个黑盒加权分数，因为权重容易变成不可解释调参。建议用两层排序：

1. Pareto 过滤：如果玩法 A 在安全、体力、时间、费用、证据数上均不差于玩法 B，并至少一项更好，则 B 被支配并移除。
2. 字典序排序：按固定优先级排序，而不是训练或调参。

排序键示例：

```text
(
  decision_rank,  # pass < conditional < unknown < fail
  hard_fail,
  missing_required_evidence_count,
  unresolved_risk_count,
  required_action_count,
  total_cost_cny_if_known,
  duration_max_min,
  -evidence_count
)
```

每个候选都能解释：

```text
推荐：天子山索道 + 袁家界轻松线
原因：
1. 当前用户画像包含老人和低龄儿童，适配评估结果为 conditional/pass。
2. 帖子证据显示可使用索道/电梯替代长距离爬升。
3. 999 级台阶方案不会作为默认选择；若使用穿山扶梯需增加 32 元。
4. 有 3 条帖子证据支持，缺失证据少于其他候选。
```

## 6. 写图建议

### 6.1 约束与索引

```cypher
CREATE CONSTRAINT post_id IF NOT EXISTS
FOR (p:Post) REQUIRE p.id IS UNIQUE;

CREATE CONSTRAINT place_key IF NOT EXISTS
FOR (p:Place) REQUIRE (p.name, p.run_id) IS UNIQUE;

CREATE CONSTRAINT route_variant_id IF NOT EXISTS
FOR (rv:RouteVariant) REQUIRE rv.id IS UNIQUE;

CREATE CONSTRAINT play_mode_id IF NOT EXISTS
FOR (pm:PlayMode) REQUIRE pm.id IS UNIQUE;

CREATE INDEX constraint_lookup IF NOT EXISTS
FOR (c:Constraint) ON (c.metric, c.value_num, c.unit, c.run_id);
```

### 6.2 写入帖子和玩法

```cypher
MERGE (post:Post {id: $post_id})
SET post.run_id = $run_id,
    post.title = $title,
    post.author = $author,
    post.query = $query,
    post.source_file = $source_file,
    post.result_index = $result_index;

MERGE (place:Place {name: $place_name, run_id: $run_id})
SET place.aliases = $aliases;

MERGE (rv:RouteVariant {id: $route_variant_id})
SET rv.run_id = $run_id,
    rv.name = $route_name,
    rv.destination = $destination,
    rv.style = $style,
    rv.physical_load_rank = $physical_load_rank,
    rv.duration_min = $duration_min,
    rv.duration_max_min = $duration_max_min,
    rv.cost_min_cny = $cost_min_cny,
    rv.cost_max_cny = $cost_max_cny;

MERGE (post)-[:DESCRIBES]->(rv)
MERGE (rv)-[:IN_PLACE]->(place);
```

### 6.3 写入底层事实、风险缓解和证据

```cypher
MERGE (c:Constraint {
  id: $constraint_id
})
SET c.run_id = $run_id,
    c.metric = $metric,
    c.value_num = $value_num,
    c.value_text = $value_text,
    c.unit = $unit,
    c.bound = $bound,
    c.polarity = $polarity;

MERGE (ev:Evidence {id: $evidence_id})
SET ev.run_id = $run_id,
    ev.text = $evidence_text,
    ev.source_file = $source_file,
    ev.result_index = $result_index;

MATCH (rv:RouteVariant {id: $route_variant_id})
MERGE (rv)-[:HAS_CONSTRAINT]->(c)
MERGE (c)-[:SUPPORTED_BY]->(ev);
```

`Requirement/Risk/Mitigation` 也采用同样模式写入：先写 evidence，再把结构化事实连接到 `RouteVariant`，并用 `SUPPORTED_BY` 指向证据。这样后续 `FitAssessment` 能明确说明每个判断来自哪段小红书文本。

## 7. 对当前代码的最小改造路径

1. 在 `add_autoglm_result_to_mem0()` 中保留原始文本写向量库，但使用 graph-disabled client 或直接走 vector-only 写入，避免继续用 mem0 的自由实体关系写法污染 Neo4j。
2. 新增 `parse_autoglm_posts(json_path)`：把 48 条 result 拆为 `PostEvidence`，去掉任务日志和平台元信息。
3. 新增 `extract_travel_facts(post)`：输出严格 schema 的 `RouteVariant/Constraint/Requirement/Risk/Mitigation/Evidence`。
4. 新增 `write_travel_graph(facts, run_id="xhs")`：使用白名单 Cypher 写入 `Post/Place/RouteVariant/Constraint/Requirement/Risk/Mitigation/Evidence`。
5. 新增 `cluster_play_modes(destination, run_id="xhs")`：读取同目的地的 `RouteVariant`，在应用侧用 NetworkX Louvain 或服务侧 Leiden/GDS 生成 `PlayMode`。
6. 新增 `parse_traveler_profile(user_text)`：把用户输入解析为 `TravelerProfile`，只描述画像和偏好，不直接生成大量活动规则。
7. 新增 `evaluate_fit(profile, route_evidence)`：基于 `Requirement/Risk/Mitigation/Evidence` 输出结构化 `FitAssessment`。
8. 新增 `query_play_modes(profile, destination)`：召回候选玩法，运行证据适配评估，再 Pareto + 字典序排序，并返回证据链。

## 8. 为什么这比单纯优化 prompt 更可靠

1. Prompt 只能提高抽取概率，不能保证图结构合法；schema 校验和关系白名单能保证非法关系不落库。
2. 当前图中的自环、label 漂移、作者/hashtag 污染不是 prompt 单点问题，而是缺少写图前的确定性过滤。
3. 社区发现把多帖子、多路线、多证据归并成玩法簇，能表达“同一景点有多种玩法”，这是单条关系抽取做不到的。
4. 画像适配评估让“老人、孩子、省钱、轻松游”与帖子证据中的能力要求、风险、缓解措施对齐，而不是堆砌具体活动 `if/else`。
5. Pareto 和字典序排序可解释、可复现，不依赖训练和调参，适合先做工程落地。

## 9. 验收标准

1. 查询 `run_id="xhs"` 时，不再出现作者、hashtag、平台日志作为旅游主图节点。
2. 主图中没有有效自环。
3. 对张家界类帖子，能产生多个 `RouteVariant/PlayMode`，例如“森林公园两日亲子线”“天门山 B 线扶梯线”“金鞭溪轻松玩水线”“高端包车定制线”。
4. 每个候选玩法至少有一条 `Evidence` 可回溯到原始帖子。
5. 用户输入“70 岁老人、12 岁和 5 岁儿童、经济、省钱、轻松游”后，系统能生成 `TravelerProfile`，并对候选玩法生成 `FitAssessment`。
6. 输出推荐时能解释过滤原因，例如“999 级台阶对当前画像风险高；若选择 32 元扶梯替代则该路段可保留”。
7. 对“特种兵打卡”类帖子，系统能整体过滤高强度玩法，同时保留其中低强度、低成本、可替代交通的片段信息。
8. 对“海边冲浪 + 5 岁儿童”这类新活动，如果帖子缺少儿童安全证据，系统应输出 `unknown` 或 `fail`，而不是靠手写活动规则武断判断。

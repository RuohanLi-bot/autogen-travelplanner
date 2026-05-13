# Current Mainline Flow

本文档梳理当前 `travel-planner` 中，从用户输入到最终行程输出的程序逻辑主线。这里描述的是当前代码实现，不是理想设计稿。

## 1. 主入口

主入口在 `main_mem0.py` 的 `run_xhs_full_itinerary_flow(...)`。

这条主线的目标是：

1. 拿到用户 query 对应的完整 `TravelerProfile`
2. 基于本地图谱筛出候选 `PlayMode`
3. 让候选玩法针对当前画像做动态量化
4. 对候选玩法打分
5. 在 beam search 中组合成多日 `ItinerarySkeleton`
6. 渲染最终行程并校验

## 2. 核心数据对象

### 2.1 TravelerProfile

定义在 `xhs_travel_graph/models.py`。

当前它是整个主线唯一的用户侧主状态对象，关键字段有：

- `profile_id`
- `destination`
- `user_query`
- `figure`
- `budget`
- `strength`
- `activity`
- `preference`
- `source`

其中：

- `figure` 是规范化的人群标签列表，例如 `adult`、`senior:70:normal`
- 其余四个维度都是 `List[dict]`
- 每个 `dict` 表示一条约束或偏好 spec

当前 spec 的最小 schema 是：

```python
{
    "metric_key": str,
    "dimension": "budget" | "strength" | "activity" | "preference",
    "op": "<=" | ">=" | "==" | "in" | "not_in",
    "value": int | float | str | List[str] | None,
    "description": str,
    "hard": bool,
}
```

### 2.2 PlanningBudget

定义在 `xhs_constraints/models.py`。

当前它不再承载完整能力边界翻译链，而主要承载规划旋钮和评分偏好，例如：

- `budget_level`
- `require_rest_buffer`
- `avoid_cross_scenic_area`
- `max_core_places_per_day`
- `preferred_candidate_tags`
- `weights`

### 2.3 PlayModeFit

定义在 `xhs_constraints/models.py`。

当前 `PlayModeFit` 有两层含义：

1. 保留了兼容旧逻辑的 `cost_vector`
2. 新增了针对当前画像的 `constraint_projection`

现在真正参与约束比较的是：

- `constraint_projection`

不是旧的固定 `cost_vector`。

### 2.4 SkeletonDay

定义在 `xhs_constraints/models.py`。

它表示优化器拼出来的一天骨架，当前新增了：

- `projected_metrics`

这个字段用于把当天采用的玩法量化结果带到后续 optimizer 和 validator。

## 3. 主线步骤

## 3.1 解析用户画像 seed

函数：

- `xhs_travel_graph/profile_parser.py::parse_traveler_profile`

作用：

- 调用 LLM，从 `user_query` 抽取 `figure`
- 如果 query 里显式出现预算、活动、偏好信号，也会生成少量 seed specs
- 构造一个 `TravelerProfile`

这一步输出的是 `source="query_seed"` 的初始画像。

## 3.2 复用图中已有画像

调用：

- `xhs_constraints/capability_graph_writer.py::load_traveler_profile_by_figure`

逻辑：

- 按 `TravelerProfile.figure` 查 Neo4j
- 如果命中，直接加载完整 `TravelerProfile`
- 这意味着四个维度已经是之前求证并存图后的完整 specs

命中后：

- `source` 会变成 `graph_reuse`
- 跳过后续 specs 求证步骤

## 3.3 图未命中时生成并求证 specs

### 3.3.1 生成 specs

函数：

- `xhs_constraints/query_semantics.py::generate_figure_mapping_questions`

虽然函数名还叫 `generate_figure_mapping_questions`，但当前实际返回的是：

- `List[dict]` 的约束/偏好 specs

输入来源：

- `TravelerProfile.figure`
- `TravelerProfile.destination`
- `TravelerProfile.user_query`

输出内容：

- budget、strength、activity、preference 四个维度中，后续值得验证的 specs

### 3.3.2 逐条调用 AutoGLM 求证

主线在 `main_mem0.py` 中对每个 spec 循环调用 AutoGLM。

当前输入给 AutoGLM 的核心内容是：

- `traveler_profile.figure`
- `spec.description`

目的是让 AutoGLM 去小红书里搜索这条 spec 对应的可计算或可判别值。

### 3.3.3 把求证结果回填到 spec

函数：

- `xhs_constraints/query_semantics.py::ground_constraint_spec_from_raw_result`

作用：

- 把 AutoGLM 的自然语言输出再交给 LLM
- 只抽取当前 spec 对应的最终 `value`

### 3.3.4 把 grounded specs 回填到 TravelerProfile

函数：

- `xhs_constraints/query_semantics.py::apply_constraint_specs_to_profile`

作用：

- 按 `dimension` 分组
- 回填到：
  - `TravelerProfile.budget`
  - `TravelerProfile.strength`
  - `TravelerProfile.activity`
  - `TravelerProfile.preference`

这一步完成后，得到的是：

- `source="grounded"` 的完整画像

### 3.3.5 写回图

函数：

- `xhs_constraints/capability_graph_writer.py::write_traveler_profile`

当前图模型只存：

- `TravelerProfile`
- `Budget`
- `Strength`
- `Activity`
- `Preference`

每个维度节点直接存对应的原始 `List[dict]`。

## 3.4 基于完整画像匹配候选玩法簇

函数：

- `xhs_travel_graph/matcher.py::query_matching_play_modes`

作用：

- 从本地图谱中取出目的地下的候选 `PlayMode`
- 先做一轮与当前 `TravelerProfile` 的粗匹配

这一步更偏“候选集筛选”，不是最终行程生成。

## 3.5 从画像派生规划旋钮

函数：

- `xhs_constraints/constraint_calibrator.py::build_planning_budget`

当前这个文件已经不再承担旧的能力估计翻译链，而只负责从画像派生一些规划旋钮，例如：

- 是否需要休息缓冲
- 是否避免跨景区系统
- 每天最多几个核心点
- 偏好的 candidate tags
- 预算等级

它的输出是 `PlanningBudget`。

## 3.6 让候选玩法针对当前画像做动态量化

函数：

- `xhs_constraints/playmode_fits.py::build_play_mode_fits`

这是当前主线里很关键的一步。

当前逻辑不是直接使用固定的 `cost_vector` 来表达玩法，而是：

1. 先拿到玩法簇相关的原始证据
2. 根据当前 `TravelerProfile` 中真正出现的 specs
3. 为每个候选玩法生成当前画像下的 `constraint_projection`

也就是说：

- `TravelerProfile` 定义了“当前要衡量什么”
- 候选玩法再动态产出这些 metric 对应的值

当前 `cost_vector` 还保留着，但只是兼容层和次级说明，不再是主约束来源。

## 3.7 按画像 specs 直接打分

函数：

- `xhs_constraints/scorer.py::score_play_modes`

当前 scorer 的核心逻辑已经改成：

1. 遍历 `TravelerProfile` 中全部 specs
2. 在每个 `PlayModeFit.constraint_projection` 里找对应值
3. 按 spec 的 `op`、`value`、`hard` 逐条比较

判断规则大致是：

- `hard=True` 且不满足：记 `hard_violation`
- `hard=False` 且不满足：记 `soft_violation`
- activity/preference 的 `in/not_in/==` 也在这一层比较

最后综合得到：

- `total_score`
- `hard_violations`
- `soft_violations`

## 3.8 用 beam search 组合成多日行程骨架

函数：

- `xhs_constraints/optimizer.py::optimize_itinerary_from_play_modes`

核心状态对象：

- `_PlanState`

当前 `_PlanState` 累积的是：

- `days`
- `used_places`
- `used_play_modes`
- `trip_numeric_totals`
- `presence_values`
- `score`

这一步的逻辑是：

1. 从没有 hard violation 的候选玩法出发
2. 逐天扩展 beam
3. 每加入一个玩法日，就把它的 `projected_metrics` 累加进 `_PlanState`
4. 检查 itinerary-level hard specs 是否被违反
5. 如果违反则剪枝，否则进入下一轮 beam

如果某一天完全扩不出可行候选：

- 退化为一个 `Rest` 缓冲日

输出是：

- `ItinerarySkeleton`

## 3.9 从骨架渲染最终行程

函数：

- `xhs_constraints/final_writer.py::write_final_itinerary`

作用：

- 根据 `ItinerarySkeleton` 确定性生成最终 `Itinerary`

也就是说，此时 LLM 已经不在最终行程文本生成里起主导作用，更多是：

- 上游画像抽取
- specs 生成
- specs 求证
- 候选玩法动态量化

## 3.10 最终校验

函数：

- `xhs_constraints/validator.py::validate_final_itinerary`

当前 validator 除了做结构性校验，还会回放 `TravelerProfile` 的 hard specs。

它主要检查：

- 最终骨架/行程是否违反 hard 数值限制
- 是否违反 hard 的 activity/preference 约束
- 是否有结构性问题，例如骨架和最终文本不一致

## 4. 当前主线里的角色分工

### `TravelerProfile`

唯一用户侧主状态对象。

### `PlanningBudget`

不是画像本身，而是由画像派生出的 planner knobs。

### `PlayModeFit.constraint_projection`

候选玩法针对当前画像动态量化后的结果。

### `ScoredPlayMode`

约束比较和综合打分后的候选玩法。

### `ItinerarySkeleton`

多日优化结果，是真正的“行程生成中间产物”。

### `Itinerary`

最终用户看到的行程文本结构。

## 5. 当前主线的关键特点

### 5.1 已经完成的收口

当前主线已经不再依赖这些旧链路：

- `ProfileSignature`
- `CapabilityQuestion`
- `CapabilityObservation`
- `CapabilityEstimate`
- 旧的 `MetricLimit -> planning_budget_to_constraints` 主链

### 5.2 当前仍然保留的兼容层

虽然主线已经切到 `TravelerProfile -> projection -> score -> optimize`，但仍然保留：

- `PlayModeCostVector`

它目前还存在于 `PlayModeFit` 中，但不再是主约束来源。

## 6. 一句话总结

当前程序主线可以概括为：

1. 从 query 抽出 `TravelerProfile`
2. 优先按 `figure` 复用图中的完整画像
3. 图未命中时，生成 specs 并逐条用 AutoGLM 去小红书求证
4. 得到完整 `TravelerProfile`
5. 查询候选 `PlayMode`
6. 让候选玩法针对当前画像动态产出 `constraint_projection`
7. 直接按画像 specs 打分
8. 在 beam search 中累积这些指标并组合出 `ItinerarySkeleton`
9. 渲染最终行程
10. 回放 hard specs 做最终校验

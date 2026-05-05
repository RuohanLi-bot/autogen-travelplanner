from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

from .models import XHSPostEvidence


TITLE_PATTERNS = [
    re.compile(r"\*\*(?:笔记)?标题[:：]\*\*\s*(.+)"),
    re.compile(r"\*\*(?:笔记)?标题\*\*[:：]\s*(.+)"),
    re.compile(r"标题[:：]\s*(.+)"),
]

AUTHOR_PATTERNS = [
    re.compile(r"\*\*(?:笔记)?作者[:：]\*\*\s*(.+)"),
    re.compile(r"\*\*(?:笔记)?作者\*\*[:：]\s*(.+)"),
    re.compile(r"作者[:：]\s*(.+)"),
]

BODY_MARKERS = [
    "**正文内容：**",
    "**正文内容:**",
    "**笔记内容概要**",
    "**笔记内容概要：**",
    "**第2条帖子完整正文内容如下：**",
]

NOISE_PREFIXES = (
    "任务已完成",
    "我已经成功搜索",
    "我成功搜索了",
    "笔记内容已完整展示",
    "根据任务要求",
)


def load_autoglm_posts(json_path: Path, run_id: str = "xhs") -> List[XHSPostEvidence]:
    json_path = Path(json_path).expanduser().resolve()
    with json_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    items = payload if isinstance(payload, list) else [payload]
    posts = []
    total = len(items)
    for idx, item in enumerate(items):
        if not isinstance(item, dict):
            continue
        result = item.get("result")
        if not isinstance(result, str) or not result.strip():
            continue
        posts.append(
            parse_autoglm_result_item(
                item=item,
                source_file=json_path,
                result_index=idx,
                result_count=total,
                run_id=run_id,
            )
        )
    return posts


def parse_autoglm_result_item(
    *,
    item: Dict,
    source_file: Path,
    result_index: int,
    result_count: int,
    run_id: str = "xhs",
) -> XHSPostEvidence:
    raw_result = str(item.get("result") or "")
    body, meta = clean_autoglm_result(raw_result)
    task = str(item.get("task") or "")
    query = _extract_query(task) or _extract_query(raw_result)
    title = meta.get("title", "")
    author = meta.get("author", "")
    post_id = _stable_id(str(source_file), result_index, title, author, body[:120])
    parse_quality = _parse_quality(body=body, title=title)
    return XHSPostEvidence(
        post_id=post_id,
        run_id=run_id,
        source_file=str(source_file),
        result_index=result_index,
        result_count=result_count,
        task=task,
        query=query,
        title=title,
        author=author,
        body=body,
        raw_result=raw_result,
        parse_quality=parse_quality,
    )


def clean_autoglm_result(result: str) -> Tuple[str, Dict[str, str]]:
    text = _normalize_lines(result)
    title = _first_match(TITLE_PATTERNS, text)
    author = _first_match(AUTHOR_PATTERNS, text)
    body = _slice_body(text)
    body = _drop_noise_lines(body)
    body = _truncate_at_markers(body, ["**评论数", "评论数：", "笔记内容已完整展示"])
    body = body.strip()
    if not body:
        body = _drop_noise_lines(text).strip()
    return body, {"title": title, "author": author}


def _normalize_lines(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [" ".join(line.strip().split()) for line in text.splitlines()]
    return "\n".join(line for line in lines if line)


def _first_match(patterns: List[re.Pattern], text: str) -> str:
    for pattern in patterns:
        match = pattern.search(text)
        if match:
            return _clean_inline_value(match.group(1))
    return ""


def _slice_body(text: str) -> str:
    for marker in BODY_MARKERS:
        pos = text.find(marker)
        if pos >= 0:
            return text[pos + len(marker) :].strip()
    match = re.search(r"\*\*(?:正文内容|笔记内容概要)[:：]?\*\*", text)
    if match:
        return text[match.end() :].strip()
    return text


def _drop_noise_lines(text: str) -> str:
    out = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if any(stripped.startswith(prefix) for prefix in NOISE_PREFIXES):
            continue
        if re.match(r"\*\*第\d+条帖子完整正文内容如下[:：]?\*\*", stripped):
            continue
        if re.match(r"\*\*第\d+条笔记的完整正文内容[:：]?\*\*", stripped):
            continue
        out.append(stripped)
    return "\n".join(out)


def _truncate_at_markers(text: str, markers: List[str]) -> str:
    cut = len(text)
    for marker in markers:
        pos = text.find(marker)
        if pos >= 0:
            cut = min(cut, pos)
    return text[:cut]


def _extract_query(text: str) -> str:
    if not text:
        return ""
    match = re.search(r"[\"“](.+?)[\"”]", text)
    if match:
        return _clean_inline_value(match.group(1))
    match = re.search(r"搜索(.+?)(?:，|,|。|$)", text)
    if match:
        return _clean_inline_value(match.group(1))
    return ""


def _clean_inline_value(value: str) -> str:
    return value.strip().strip("*").strip().strip("：:")


def _stable_id(*parts: object) -> str:
    raw = "|".join(str(part) for part in parts)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def _parse_quality(*, body: str, title: str) -> str:
    if title and len(body) >= 120:
        return "high"
    if len(body) >= 80:
        return "medium"
    return "low"

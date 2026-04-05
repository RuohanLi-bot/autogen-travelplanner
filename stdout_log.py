"""
一次完整运行：仅一个 .log 文件。

- print（stdout）与所有 logging 记录（INFO+）写入同一文件句柄；
- 配置时会清空 root 与各已存在子 logger 上的 handlers，避免第三方库为「每次调用」
  挂上 FileHandler 导致多文件或重复写；
- WARNING 及以上同时写入 stderr（终端）。
"""
from __future__ import annotations

import logging
import sys
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Iterator, TextIO


class _FlushStreamHandler(logging.StreamHandler):
    """与 print 共用文件时，每条日志后立即 flush。"""

    def emit(self, record: logging.LogRecord) -> None:
        super().emit(record)
        self.flush()


def _reset_named_loggers_to_propagate_root() -> None:
    """
    去掉各子 logger 上自带的 handlers（常见于库在 import 或首次 log 时挂 FileHandler），
    并设 propagate=True，使所有日志统一经 root 进入「单文件」handler。
    """
    manager = logging.root.manager
    for name in list(manager.loggerDict.keys()):
        if not name:
            continue
        ref = manager.loggerDict[name]
        if isinstance(ref, logging.PlaceHolder):
            continue
        lg = logging.getLogger(name)
        lg.handlers.clear()
        lg.propagate = True


def configure_logging_for_run(log_file: TextIO) -> None:
    """
    INFO 及以上写入 log_file（与 sys.stdout 同一文件对象）；
    WARNING 及以上同时输出到 stderr。
    """
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(logging.INFO)

    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")

    fh = _FlushStreamHandler(log_file)
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)

    sh = logging.StreamHandler(sys.stderr)
    sh.setLevel(logging.WARNING)
    sh.setFormatter(fmt)

    root.addHandler(fh)
    root.addHandler(sh)

    _reset_named_loggers_to_propagate_root()


@contextmanager
def stdout_to_log_file(log_dir: Path, prefix: str = "run") -> Iterator[Path]:
    """
    单次运行只创建一个带时间戳的 .log；进程内所有 print 与 logging 均写入该文件。
    """
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    path = log_dir / f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    print(f"[运行日志 → 单文件] {path}", file=sys.stderr, flush=True)
    log_f: TextIO = open(path, "w", encoding="utf-8")
    old = sys.stdout
    sys.stdout = log_f
    try:
        configure_logging_for_run(log_f)
        yield path
    finally:
        root = logging.getLogger()
        for h in root.handlers:
            try:
                h.flush()
            except Exception:
                pass
        sys.stdout = old
        log_f.close()
        print(f"[运行结束] 完整日志（单文件）: {path}", file=sys.stderr, flush=True)

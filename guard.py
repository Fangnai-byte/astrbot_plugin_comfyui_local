"""频率限制 + 生成日志：两件「记账」的事放一起。

为什么需要频率限制
------------------
出图很贵（一张 12~25 秒、吃满显存）。群里一个人连刷就会把队列堵死，
别人全在等。所以两层限制：

* **冷却**：同一个人两次请求之间至少隔 ``rate_limit_cooldown`` 秒 —— 挡住
  「手抖连点两下」这种。
* **配额**：``rate_limit_window`` 秒内最多 ``rate_limit_count`` 次 —— 挡住
  「一口气刷十张」。

键用**发送者 ID**（不是会话）：同一个人在私聊和群里共享配额，换群刷没用。

为什么需要生成日志
------------------
回执要短（用户明确要求），但「什么时候画的、用了什么、跑多久、出来安不安全」
又必须能查。所以细节全部落到 ``draw.log``，聊天里只回一句。日志按行写，
超过大小就切成 ``draw.log.1``，不会无限长。
"""

from __future__ import annotations

import pathlib
import time
from collections import deque
from dataclasses import dataclass, field


@dataclass
class RateLimitResult:
    """一次配额检查的结果。"""

    allowed: bool
    #: 不允许时，人类可读的原因（可直接发给用户）
    reason: str = ""
    #: 还要等多少秒（0 表示不用等）
    retry_after: float = 0.0
    #: 本窗口内已用次数
    used: int = 0
    #: 本窗口的配额
    limit: int = 0


class RateLimiter:
    """滑动窗口配额 + 最小间隔冷却。纯内存，重启即清空（够用，不必持久化）。"""

    def __init__(self, count: int, window: float, cooldown: float) -> None:
        self.count = max(0, int(count))
        self.window = max(0.0, float(window))
        self.cooldown = max(0.0, float(cooldown))
        #: key -> 最近的请求时间戳（时间升序）
        self._hits: dict[str, deque[float]] = {}

    def _prune(self, key: str, now: float) -> deque[float]:
        hits = self._hits.setdefault(key, deque())
        if self.window > 0:
            cutoff = now - self.window
            while hits and hits[0] < cutoff:
                hits.popleft()
        return hits

    def check(self, key: str) -> RateLimitResult:
        """检查并**消耗**一次配额。允许时把当前时间记进窗口。"""
        now = time.monotonic()
        hits = self._prune(key, now)

        # 冷却优先报：用户刚点过，等两秒比「配额用完了」更贴近实际
        if self.cooldown > 0 and hits:
            waited = now - hits[-1]
            if waited < self.cooldown:
                left = self.cooldown - waited
                return RateLimitResult(
                    allowed=False,
                    reason=f"刚画过一张，等 {left:.0f} 秒再来～",
                    retry_after=left,
                    used=len(hits),
                    limit=self.count,
                )

        if self.count > 0 and len(hits) >= self.count:
            # 窗口是滑动的：最早那次过期就能再来
            left = (hits[0] + self.window) - now if self.window > 0 else 0.0
            left = max(0.0, left)
            minutes = self.window / 60.0
            window_text = (
                f"{minutes:.0f} 分钟" if minutes >= 1 else f"{self.window:.0f} 秒"
            )
            return RateLimitResult(
                allowed=False,
                reason=(
                    f"{window_text}内已经画了 {len(hits)} 张（上限 {self.count} 张），"
                    f"等 {_human(left)} 再来～"
                ),
                retry_after=left,
                used=len(hits),
                limit=self.count,
            )

        hits.append(now)
        return RateLimitResult(
            allowed=True, used=len(hits), limit=self.count
        )


def _human(seconds: float) -> str:
    seconds = max(0.0, seconds)
    if seconds < 60:
        return f"{seconds:.0f} 秒"
    minutes = seconds / 60.0
    if minutes < 60:
        return f"{minutes:.0f} 分钟"
    return f"{minutes / 60:.1f} 小时"


@dataclass
class DrawRecord:
    """一次出图的完整记录。字段顺序就是日志里的列顺序。"""

    status: str = "ok"
    user: str = ""
    chat: str = ""
    request: str = ""
    preset: str = ""
    workflow: str = ""
    reference: str = ""
    mode: str = ""
    denoise: str = ""
    size: str = ""
    seed: str = ""
    elapsed: str = ""
    output: str = ""
    exposure: str = ""
    note: str = ""


class DrawLog:
    """按行追加的生成日志，带大小轮转。

    一行一条记录，``键=值`` 用 ``|`` 分隔，方便人看也方便 grep。
    值里的 ``|`` 和换行会被替换掉，避免破坏列结构。
    """

    def __init__(self, path: pathlib.Path, max_bytes: int = 512 * 1024) -> None:
        self.path = path
        self.max_bytes = max(0, int(max_bytes))

    def _rotate(self) -> None:
        if self.max_bytes <= 0:
            return
        try:
            if self.path.exists() and self.path.stat().st_size >= self.max_bytes:
                backup = self.path.with_suffix(self.path.suffix + ".1")
                if backup.exists():
                    backup.unlink()
                self.path.rename(backup)
        except OSError:
            pass  # 轮转失败也要继续写，不能因为日志把出图搞崩

    def write(self, record: DrawRecord) -> None:
        fields = [
            time.strftime("%Y-%m-%d %H:%M:%S"),
            record.status,
            f"user={record.user}",
            f"chat={record.chat}",
            f"req={record.request}",
            f"preset={record.preset}",
            f"wf={record.workflow}",
            f"ref={record.reference}",
            f"mode={record.mode}",
            f"denoise={record.denoise}",
            f"size={record.size}",
            f"seed={record.seed}",
            f"elapsed={record.elapsed}",
            f"out={record.output}",
            f"exposure={record.exposure}",
            f"note={record.note}",
        ]
        line = " | ".join(_clean(f) for f in fields) + "\n"
        self._rotate()
        try:
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(line)
        except OSError:
            pass

    def tail(self, lines: int = 15) -> list[str]:
        """读最后若干行，给「/画图 日志」用。"""
        if not self.path.exists():
            return []
        try:
            with self.path.open("r", encoding="utf-8", errors="replace") as fh:
                content = fh.readlines()
        except OSError:
            return []
        return [ln.rstrip("\n") for ln in content[-max(1, lines):]]


def _clean(value: object) -> str:
    text = str(value if value is not None else "")
    return text.replace("|", "/").replace("\n", " ").replace("\r", " ").strip()

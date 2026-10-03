"""ComfyUI HTTP + WebSocket 客户端。

为什么走 WebSocket 而不是轮询 ``/history``
------------------------------------------
ComfyUI 的 ``/history`` 依赖它的 SQLite 库；库一但起不来（例如
``user/comfyui.db.lock`` 被占用或没有写权限），``/history/<id>`` 会**一直是空的**，
于是轮询永远等不到结果、只能超时。WebSocket 是 ComfyUI 官方的推送通道：
执行进度、每个节点产出、以及异常栈都直接推过来，比轮询更快也更可靠。
"""

from __future__ import annotations

import asyncio
import base64
import json
import random
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlencode

import aiohttp


class ComfyError(Exception):
    """与 ComfyUI 通信或执行失败。"""


@dataclass
class ProgressEvent:
    """一次进度推送。"""

    kind: str  # queued | cached | progress | executing | output | done | error
    node: str | None = None
    value: int = 0
    maximum: int = 0
    message: str = ""
    outputs: dict[str, Any] = field(default_factory=dict)

    @property
    def percent(self) -> int:
        if self.maximum <= 0:
            return 0
        return max(0, min(100, round(self.value * 100 / self.maximum)))


@dataclass
class GenerationResult:
    """一次生成的结果。"""

    prompt_id: str
    images: list[dict[str, Any]] = field(default_factory=list)
    videos: list[dict[str, Any]] = field(default_factory=list)
    texts: list[str] = field(default_factory=list)
    duration: float = 0.0

    @property
    def empty(self) -> bool:
        return not (self.images or self.videos or self.texts)


_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"}
_VIDEO_EXTS = {".mp4", ".webm", ".mkv", ".mov", ".avi"}


class ComfyClient:
    """一个 ComfyUI 服务端的轻量客户端。"""

    def __init__(
        self,
        server: str,
        *,
        timeout: int = 300,
        client_id: str | None = None,
    ) -> None:
        self.server = self._normalize(server)
        self.timeout = timeout
        self.client_id = client_id or str(uuid.uuid4())
        self._session: aiohttp.ClientSession | None = None
        self._object_info: dict[str, Any] | None = None

    @staticmethod
    def _normalize(server: str) -> str:
        text = (server or "").strip().rstrip("/")
        if not text:
            raise ComfyError("没配置 ComfyUI 地址")
        if not text.startswith(("http://", "https://")):
            text = f"http://{text}"
        return text

    @property
    def ws_url(self) -> str:
        base = self.server.replace("https://", "wss://", 1).replace("http://", "ws://", 1)
        return f"{base}/ws"

    async def _ensure_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=None, sock_connect=15)
            )
        return self._session

    async def close(self) -> None:
        if self._session is not None and not self._session.closed:
            await self._session.close()
        self._session = None

    # ------------------------------------------------------------------ HTTP

    async def _get_json(self, path: str, timeout: float = 30) -> Any:
        session = await self._ensure_session()
        try:
            async with session.get(
                f"{self.server}{path}", timeout=aiohttp.ClientTimeout(total=timeout)
            ) as resp:
                if resp.status != 200:
                    body = await resp.text()
                    raise ComfyError(f"GET {path} 返回 {resp.status}：{body[:300]}")
                return await resp.json(content_type=None)
        except aiohttp.ClientError as exc:
            raise ComfyError(f"连不上 ComfyUI（{self.server}）：{exc}") from exc
        except asyncio.TimeoutError as exc:
            raise ComfyError(f"请求 {path} 超时") from exc

    async def object_info(self, *, refresh: bool = False) -> dict[str, Any]:
        """节点定义表（带缓存），UI→API 转换和模型枚举都要用。"""
        if self._object_info is None or refresh:
            self._object_info = await self._get_json("/object_info", timeout=120)
        return self._object_info

    async def checkpoints(self) -> list[str]:
        info = await self.object_info()
        try:
            return list(info["CheckpointLoaderSimple"]["input"]["required"]["ckpt_name"][0])
        except (KeyError, IndexError, TypeError):
            return []

    async def system_stats(self) -> dict[str, Any]:
        return await self._get_json("/system_stats")

    async def queue_state(self) -> dict[str, int]:
        data = await self._get_json("/queue", timeout=15)
        return {
            "running": len(data.get("queue_running") or []),
            "pending": len(data.get("queue_pending") or []),
        }

    #: 预检用的 1x1 透明 PNG（base64）。
    _PROBE_PNG_B64 = (
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8DwHwAF"
        "BQIAX8jx0gAAAABJRU5ErkJggg=="
    )

    @property
    def _probe_png(self) -> bytes:
        return base64.b64decode(self._PROBE_PNG_B64)

    async def check_image_write(self) -> str:
        """预检 ComfyUI 能否把图片写到磁盘，返回一句问题描述；没问题返回空串。

        做法是往 ``/upload/image`` 上传一张 1x1 的 PNG —— 它和 ``SaveImage``
        一样要落到 ComfyUI 的数据目录，权限不足或磁盘满会同样失败。

        为什么值得做这个预检：出图要跑几十秒，而写盘失败只在**最后一步**的
        ``SaveImage`` 才暴露，用户会白等一分钟然后收到一句难懂的
        ``PermissionError``。提前几秒钟问一句，就能直接说清原因。
        """
        session = await self._ensure_session()
        form = aiohttp.FormData()
        form.add_field(
            "image",
            self._probe_png,
            filename="_probe.png",
            content_type="image/png",
        )
        form.add_field("overwrite", "true")
        try:
            async with session.post(
                f"{self.server}/upload/image",
                data=form,
                timeout=aiohttp.ClientTimeout(total=15),
            ) as resp:
                if resp.status == 200:
                    return ""
                body = await resp.text()
                return f"ComfyUI 无法写入图片（HTTP {resp.status}）：{body[:200]}"
        except (aiohttp.ClientError, asyncio.TimeoutError):
            # 网络层问题交给正常流程报错，这里只关心写盘
            return ""

    async def upload_image(self, name: str, blob: bytes) -> str:
        """把图片传到 ComfyUI 的 input 目录，返回可给 LoadImage 用的文件名。"""
        session = await self._ensure_session()
        form = aiohttp.FormData()
        form.add_field("image", blob, filename=name, content_type="image/png")
        form.add_field("overwrite", "true")
        try:
            async with session.post(
                f"{self.server}/upload/image",
                data=form,
                timeout=aiohttp.ClientTimeout(total=120),
            ) as resp:
                if resp.status != 200:
                    body = await resp.text()
                    raise ComfyError(f"上传参考图失败（HTTP {resp.status}）：{body[:200]}")
                data = await resp.json(content_type=None)
        except aiohttp.ClientError as exc:
            raise ComfyError(f"上传参考图失败：{exc}") from exc
        # ComfyUI 会把它放进 input/，同名时返回实际名字
        return data.get("name") or name

    async def download(self, item: dict[str, Any], timeout: float = 120) -> bytes:
        """按 /view 把产出的图片、视频取回来。"""
        query = urlencode(
            {
                "filename": item.get("filename", ""),
                "subfolder": item.get("subfolder", "") or "",
                "type": item.get("type", "output") or "output",
            }
        )
        session = await self._ensure_session()
        try:
            async with session.get(
                f"{self.server}/view?{query}",
                timeout=aiohttp.ClientTimeout(total=timeout),
            ) as resp:
                if resp.status != 200:
                    body = await resp.text()
                    raise ComfyError(f"下载 {item.get('filename')} 失败：{resp.status} {body[:200]}")
                return await resp.read()
        except aiohttp.ClientError as exc:
            raise ComfyError(f"下载产出失败：{exc}") from exc

    # ------------------------------------------------------------- 提交/生成

    async def generate(
        self,
        prompt: dict[str, Any],
        *,
        on_event: Callable[[ProgressEvent], Any] | None = None,
    ) -> GenerationResult:
        """提交 prompt 并等待执行结束。

        调用方在 ``on_event`` 里可以拿到进度、缓存命中和节点产出。
        返回的 ``images`` / ``videos`` 是 ``/view`` 的参数（还没下载）。
        """
        import time

        session = await self._ensure_session()
        started = time.monotonic()
        result = GenerationResult(prompt_id="")

        async def emit(event: ProgressEvent) -> None:
            if on_event is None:
                return
            outcome = on_event(event)
            if asyncio.iscoroutine(outcome):
                await outcome

        try:
            async with session.ws_connect(
                f"{self.ws_url}?clientId={self.client_id}",
                heartbeat=30,
                max_msg_size=0,
            ) as ws:
                try:
                    async with session.post(
                        f"{self.server}/prompt",
                        json={"prompt": prompt, "client_id": self.client_id},
                        timeout=aiohttp.ClientTimeout(total=60),
                    ) as resp:
                        payload = await resp.json(content_type=None)
                        if resp.status != 200:
                            raise ComfyError(self._format_submit_error(payload))
                except aiohttp.ClientError as exc:
                    raise ComfyError(f"提交任务失败：{exc}") from exc

                result.prompt_id = payload.get("prompt_id", "")
                await emit(ProgressEvent(kind="queued", message=result.prompt_id))

                deadline = time.monotonic() + self.timeout
                while True:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise ComfyError(f"等待生成超时（{self.timeout} 秒）")
                    try:
                        frame = await ws.receive(timeout=min(remaining, 30))
                    except asyncio.TimeoutError:
                        continue
                    if frame.type is aiohttp.WSMsgType.TEXT:
                        data = json.loads(frame.data)
                    elif frame.type in (
                        aiohttp.WSMsgType.CLOSED,
                        aiohttp.WSMsgType.CLOSING,
                        aiohttp.WSMsgType.CLOSE,
                    ):
                        raise ComfyError("与 ComfyUI 的 WebSocket 连接被关闭")
                    elif frame.type is aiohttp.WSMsgType.ERROR:
                        raise ComfyError(f"WebSocket 出错：{ws.exception()}")
                    else:
                        continue

                    kind = data.get("type")
                    body = data.get("data") or {}

                    # /ws 会广播**所有**客户端的任务，必须按 prompt_id 过滤，
                    # 否则同一台机器上别人的任务会污染这里的进度。
                    event_prompt = body.get("prompt_id")
                    if event_prompt is not None and event_prompt != result.prompt_id:
                        continue

                    if kind == "progress":
                        await emit(
                            ProgressEvent(
                                kind="progress",
                                node=str(body.get("node")),
                                value=int(body.get("value") or 0),
                                maximum=int(body.get("max") or 0),
                            )
                        )
                    elif kind == "execution_cached":
                        await emit(
                            ProgressEvent(
                                kind="cached",
                                message=",".join(str(n) for n in body.get("nodes") or []),
                            )
                        )
                    elif kind == "executing":
                        if body.get("node") is None:
                            break
                        await emit(ProgressEvent(kind="executing", node=str(body["node"])))
                    elif kind == "executed":
                        output = body.get("output") or {}
                        collected = self._collect_output(output, result)
                        await emit(
                            ProgressEvent(
                                kind="output",
                                node=str(body.get("node")),
                                outputs=collected,
                            )
                        )
                    elif kind == "execution_error":
                        raise ComfyError(self._format_exec_error(body))
                    elif kind == "execution_interrupted":
                        raise ComfyError("任务被中断")
        finally:
            result.duration = time.monotonic() - started

        await emit(ProgressEvent(kind="done"))
        return result

    @staticmethod
    def _collect_output(output: dict[str, Any], result: GenerationResult) -> dict[str, Any]:
        """把 ``executed`` 帧里的产出归到图片/视频/文本三类。"""
        collected: dict[str, Any] = {"images": [], "videos": [], "texts": []}
        for item in output.get("images") or []:
            if not isinstance(item, dict) or "filename" not in item:
                continue
            lowered = item["filename"].lower()
            suffix = "." + lowered.rsplit(".", 1)[-1] if "." in lowered else ""
            if suffix in _VIDEO_EXTS:
                result.videos.append(item)
                collected["videos"].append(item)
            elif suffix in _IMAGE_EXTS:
                result.images.append(item)
                collected["images"].append(item)
        # 部分节点（例如文本预览/反推）把结果放在任意字符串字段里
        for key, value in output.items():
            if key == "images":
                continue
            if isinstance(value, str) and value.strip():
                result.texts.append(value)
                collected["texts"].append(value)
            elif isinstance(value, list) and value and all(isinstance(v, str) for v in value):
                joined = "\n".join(value)
                if joined.strip():
                    result.texts.append(joined)
                    collected["texts"].append(joined)
        return collected

    @staticmethod
    def _format_submit_error(payload: Any) -> str:
        """把 /prompt 的校验失败整理成人能看懂的一句话。"""
        if not isinstance(payload, dict):
            return f"提交被拒绝：{str(payload)[:300]}"
        error = payload.get("error") or {}
        lines = [f"提交被拒绝：{error.get('message') or '未知原因'}"]
        details = error.get("details")
        if details:
            lines.append(f"  {details}")
        for node_id, info in (payload.get("node_errors") or {}).items():
            class_type = info.get("class_type", "?")
            for item in info.get("errors") or []:
                detail = item.get("details") or item.get("message")
                lines.append(f"  节点 {node_id}（{class_type}）：{detail}")
        return "\n".join(lines)[:1500]

    @staticmethod
    def _format_exec_error(body: dict[str, Any]) -> str:
        """把执行期异常整理成简短提示（栈只取最后两行）。"""
        lines = [
            f"生成失败：{body.get('node_type') or body.get('node_id')} —— "
            f"{body.get('exception_type')}: {body.get('exception_message')}"
        ]
        traceback_lines = [ln for ln in (body.get("traceback") or []) if ln.strip()]
        if traceback_lines:
            lines.append("  " + traceback_lines[-1].strip()[:300])
        return "\n".join(lines)


def random_seed() -> int:
    return random.randint(0, 2**63 - 1)

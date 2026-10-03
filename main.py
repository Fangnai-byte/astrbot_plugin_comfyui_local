"""AstrBot 插件：把本地 ComfyUI 接成「聊天里一句话出图」。

功能
----
* ``/画图 <描述>`` —— 用默认工作流出图。
* ``/画图 -w 工作流名 <描述>`` —— 指定工作流出图。
* ``/画图 工作流`` / ``/工作流`` —— 列出可用工作流。
* ``/画图 信息 [工作流名]`` —— 看某个工作流的参数摘要。
* ``/画图 重载`` —— 重新扫描工作流目录（改动 JSON 后不用重载插件）。

设计要点
--------
* **工作流直接吃 ComfyUI 的「界面格式」JSON**（菜单里普通「导出」的那种）。
  提交前由 :mod:`uigraph` 转成 API 格式，用户不必手动导出 API 格式。
* 出图走 **WebSocket** 拿结果，不轮询 ``/history``（后者在数据库不可用时永远为空）。
* 生成是**后台任务**：先立刻回一句「已开始」，跑完再把图发回会话。
  这样不阻塞 AstrBot 的消息管线，其他命令照常响应。
"""

from __future__ import annotations

import asyncio
import json
import pathlib
import shutil
import time
from dataclasses import dataclass, field
from typing import Any

from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.message_components import Image, Plain
from astrbot.api.star import Context, Star, StarTools

try:  # 兼容「作为包加载」与「作为脚本加载」两种方式
    from .comfy_client import ComfyClient, ComfyError, ProgressEvent
    from . import guard as guard_mod
    from . import persona as persona_mod
    from . import presets as presets_mod
    from . import references
    from . import safety as safety_mod
    from . import uigraph
except ImportError:  # pragma: no cover
    from comfy_client import ComfyClient, ComfyError, ProgressEvent

    import guard as guard_mod
    import persona as persona_mod
    import presets as presets_mod
    import references
    import safety as safety_mod
    import uigraph


PLUGIN_NAME = "astrbot_plugin_comfyui_local"

#: 工作流目录里不是工作流的文件名。
_NON_WORKFLOW_FILES = {
    "workflow_meta.json",
    "presets.json",
    "refs.json",
}

#: 允许把产出落到插件缓存里的扩展名。
_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"}
_VIDEO_SUFFIXES = {".mp4", ".webm", ".mkv", ".mov", ".avi"}

#: 提示词里的占位符。
_PLACEHOLDER_PROMPT = "{prompt}"
_PLACEHOLDER_TAGS = "{tags}"
#: 预设的基础提示词（角色 tag / LoRA 触发词那一大串）。
_PLACEHOLDER_BASE = "{base}"


@dataclass
class Workflow:
    """一份可用的工作流。"""

    key: str
    path: pathlib.Path
    description: str = ""
    text_slots: list[str] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)
    #: 转换后的 API prompt（懒加载，也用于展示参数摘要）。
    _api: dict[str, Any] | None = None
    #: 转换失败的原因；非空表示这份工作流不可用。
    error: str = ""

    @property
    def is_api_format(self) -> bool:
        return uigraph.is_api_format(self.raw)

    @property
    def label(self) -> str:
        return self.path.stem


def _split_ids(raw: Any) -> list[str]:
    """把配置里的名单（列表或逗号/换行分隔的字符串）统一成 list[str]。"""
    if raw is None:
        return []
    if isinstance(raw, str):
        parts = raw.replace("，", ",").replace("\n", ",").split(",")
    elif isinstance(raw, (list, tuple, set)):
        parts = [str(x) for x in raw]
    else:
        parts = [str(raw)]
    return [p.strip() for p in parts if p and p.strip()]


class ComfyUILocalPlugin(Star):
    def __init__(self, context: Context, config: Any = None) -> None:
        super().__init__(context)
        self.config = config if config is not None else {}
        self.data_dir = _resolve_data_dir()
        self.workflow_dir = self.data_dir / "workflows"
        self.cache_dir = self.data_dir / "cache"
        self.refs_dir = self.data_dir / "refs"
        self.workflow_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.refs_dir.mkdir(parents=True, exist_ok=True)
        self._seed_if_empty()
        self._seed_refs_if_empty()

        self.workflows: dict[str, Workflow] = {}
        self._workflow_errors: dict[str, str] = {}
        self.presets: dict[str, presets_mod.Preset] = {}
        self.refs: dict[str, references.Reference] = {}
        self._client: ComfyClient | None = None
        self._object_info: dict[str, Any] | None = None
        #: 同一会话同时只跑一个任务；跨会话并发由信号量统一限流。
        self._session_locks: dict[str, asyncio.Lock] = {}
        self._semaphore = asyncio.Semaphore(max(1, self._int("max_concurrency", 1)))
        #: 后台生成任务强引用，防止被 GC 回收。
        self._tasks: set[asyncio.Task[Any]] = set()

        self.blocked_dir = self.data_dir / "blocked"
        self.persona_path = self.data_dir / "persona.json"
        self._persona: dict[str, list[str]] = persona_mod.default_persona()
        self._seed_persona_if_empty()
        self._load_persona()
        self._rate_limiter = guard_mod.RateLimiter(
            count=self._int("rate_limit_count", 5),
            window=float(self._int("rate_limit_window", 600)),
            cooldown=float(self._int("rate_limit_cooldown", 10)),
        )
        self._draw_log = guard_mod.DrawLog(
            self.data_dir / "draw.log",
            max_bytes=self._int("draw_log_max_kb", 512) * 1024,
        )

    # ------------------------------------------------------------ 配置读取

    def _get(self, key: str, default: Any = None) -> Any:
        try:
            return self.config.get(key, default)
        except AttributeError:
            return default

    def _bool(self, key: str, default: bool) -> bool:
        value = self._get(key, default)
        if isinstance(value, str):
            return value.strip().lower() in ("1", "true", "yes", "on", "是", "开")
        return bool(value)

    def _int(self, key: str, default: int) -> int:
        try:
            return int(self._get(key, default))
        except (TypeError, ValueError):
            return default

    def _server(self) -> str:
        return str(self._get("server", "http://127.0.0.1:8188") or "http://127.0.0.1:8188")

    def _command(self) -> str:
        return str(self._get("command", "画图") or "画图").strip() or "画图"

    def _allowed(self, event: AstrMessageEvent) -> bool:
        """群/私聊开关与黑白名单。"""
        group_id = str(event.get_group_id() or "")
        if group_id:
            if not self._bool("enable_group", True):
                return False
        elif not self._bool("enable_private", True):
            return False

        sender = str(event.get_sender_id() or "")
        blacklist = _split_ids(self._get("user_blacklist", []))
        if sender and sender in blacklist:
            return False
        whitelist = _split_ids(self._get("group_whitelist", []))
        if whitelist and group_id and group_id not in whitelist:
            return False
        return True

    # ------------------------------------------------------- 露骨内容豁免权限

    def _r18_allowed(self, event: AstrMessageEvent | None) -> bool:
        """判断当前用户是否拿到露骨内容的豁免权。

        两道门都要过：

        1. 配置里显式开启 ``enable_r18``（默认关）
        2. 发送者在 ``r18_user_whitelist`` 里（空名单表示没人能用）

        设计上**默认拒绝**，而且与「谁能用绘图」是两套独立名单。

        这是本插件唯一的放行通道：没有它，提示词门与产出图兜底对所有会话
        一视同仁地生效。**本发行版不附带任何露骨素材**，豁免权只决定
        「白名单用户私聊时不被内容门拦」，不决定能画什么。
        """
        if not self._bool("enable_r18", False):
            return False
        whitelist = _split_ids(self._get("r18_user_whitelist", []))
        if not whitelist:
            return False
        if event is None:
            return False
        sender = str(event.get_sender_id() or "")
        return bool(sender) and sender in whitelist

    # ------------------------------------------------- 人设台词

    def _seed_persona_if_empty(self) -> None:
        """首次运行时把内置台词写成人设文件，方便用户直接改。

        只在文件不存在时写；已存在就绝不覆盖 —— 用户改过的台词不能被升级冲掉。
        """
        if self.persona_path.exists():
            return
        try:
            self.persona_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "_说明": [
                    "这里定义机器人回执用的台词，按场景分组；每个场景可以写多句，随机取一句。",
                    "· start=正常开始 / pose=换姿势去了参考图 / scene=换场景去了参考图",
                    "· private_r18=私聊+白名单（可以收露骨产出） / queued=前面还有任务",
                    "· denied_prompt=提示词没过内容门 / denied_rate=触发频率限制 / blocked_image=产出图被拦",
                    "群里拒绝的台词请写得含糊些：写明白就等于当着所有人面说出对方请求了什么。",
                    "改完发「/画图 重载」生效，不用重启。删掉某个场景就用内置默认。",
                ],
                **persona_mod.default_persona(),
            }
            self.persona_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            logger.info(f"[comfyui] 已播种人设台词模板到 {self.persona_path}")
        except OSError as exc:
            logger.warning(f"[comfyui] 人设文件写入失败（用内置台词）：{exc}")

    def _load_persona(self) -> None:
        """读取人设台词。"""
        path = pathlib.Path(str(self._get("persona_file", "") or "").strip() or self.persona_path)
        persona, problem = persona_mod.load_persona(path)
        if problem:
            logger.warning(f"[comfyui] {problem}（该场景沿用内置台词）")
        self._persona = persona

    def _line(self, situation: str, **fields: Any) -> str:
        """取一句人设台词。关掉人设时返回空串，调用方自己决定兜底文案。"""
        if not self._bool("persona_enable", True):
            return ""
        return persona_mod.pick(self._persona, situation, **fields)

    async def _speak(
        self, event: AstrMessageEvent, situation: str, *, fallback: str = "", **fields: Any
    ) -> bool:
        """按场景回一句台词；没人设/取不到时用 fallback。

        抽出来是为了让每条回执都走同一条路 —— 否则很容易在某个分支里忘了套人设，
        用户就会看到画风突变的机器话（实测最容易漏的就是拒绝类分支）。
        """
        text = self._line(situation, **fields) or fallback
        if not text:
            return False
        return await self._safe_send(event, text)

    # ------------------------------------------------- 内容兜底 / 限额 / 日志

    def _nsfw_exempt(self, event: AstrMessageEvent | None) -> bool:
        """是否允许接收「可能露骨」的内容。

        条件按用户要求：**私聊** 且 **在 R18 白名单里**，两个都要满足。
        群聊一律不免（群里发出去就收不回来了），私聊但不在名单里也不免。

        复用 ``_r18_allowed`` 的两道门（``enable_r18`` + 白名单）。本发行版
        不附带任何露骨素材，所以 ``_r18_allowed`` 在这里只表示「机主允许
        白名单用户私聊免检」，与素材加载无关；两处共用同一套权限，
        不会出现「一边放行一边拦截」的自相矛盾。
        """
        if not self._bool("nsfw_allow_private_r18", True):
            return False
        if event is None:
            return False
        if str(event.get_group_id() or ""):
            return False  # 群聊：一律不放行
        return self._r18_allowed(event)

    def _check_rate(self, event: AstrMessageEvent) -> guard_mod.RateLimitResult:
        """按发送者查配额。键用 sender id：换群也共用同一份配额，换群刷没用。"""
        if not self._bool("rate_limit_enable", True):
            return guard_mod.RateLimitResult(allowed=True)
        sender = str(event.get_sender_id() or "")
        if not sender:
            # 拿不到发送者（某些适配器）就不限制，总比把所有人都拦掉强
            return guard_mod.RateLimitResult(allowed=True)
        return self._rate_limiter.check(sender)

    def _check_prompt_safety(self, text: str, event: AstrMessageEvent) -> list[str]:
        """生成**之前**筛提示词。返回命中的关键词（空表示放行）。"""
        if not self._bool("nsfw_block_prompt", True):
            return []
        if self._nsfw_exempt(event):
            return []
        return safety_mod.find_nsfw_keywords(text)

    def _nsfw_threshold(self) -> float:
        """读暴露度阈值，归一到 0~1。

        配置项是 ``int``（默认 55），因为 AstrBot 的 WebUI 对整数输入框更友好
        （float 输入框在不同版本表现不一致）。代码里统一除以 100 —— **忘了除
        就会拿 55 去比 0~1 的分值，等于这道门永远不生效**，而且不会有任何报错。
        这里兜住「用户直接填了 0.55」的情况：大于 1 才除。
        """
        raw = self._float("nsfw_skin_threshold", 55.0)
        if raw <= 0:
            return 1.0  # 填 0 视为关闭（阈值 1.0 谁都不超）
        return raw / 100.0 if raw > 1.0 else raw

    def _inspect_image(self, path: pathlib.Path) -> safety_mod.ImageVerdict | None:
        """对产出图跑一次暴露度检查。检查本身出错时返回 ``None``（不拦）。

        为什么出错不拦：判据只是兜底。图片解析失败更可能是格式怪、不是内容问题，
        因为这个把图扣下会变成「正常出图偶尔发不出去」的玄学 bug。
        """
        if not self._bool("nsfw_filter", True):
            return None
        threshold = self._nsfw_threshold()
        try:
            return safety_mod.scan_image_file(path, threshold=threshold)
        except safety_mod.SafetyError as exc:
            logger.warning(f"[comfyui] 内容检查跳过（{exc}）")
            return None

    def _quarantine(self, path: pathlib.Path) -> pathlib.Path | None:
        """把被拦下的图挪到 blocked/ 目录，返回新路径。

        为什么不直接删：判据不可靠（见 safety.py），误拦时用户还得能拿回图。
        挪出来也顺便保证它不会被误发。

        为什么也要淘汰：这里以前只进不出。判据会误拦，正常出图多了 blocked/
        就会一直长 —— 用户这台机器的 C 盘刚出过空间事故，任何「只增不减」的
        目录都是隐患。所以跟 cache/ 一样按数量留最近的。
        """
        try:
            self.blocked_dir.mkdir(parents=True, exist_ok=True)
            target = self.blocked_dir / path.name
            shutil.move(str(path), str(target))
            self._prune_blocked()
            return target
        except OSError as exc:
            logger.warning(f"[comfyui] 移动被拦图片失败：{exc}")
            return None

    def _prune_blocked(self) -> None:
        """blocked/ 超过上限时删掉最旧的。"""
        limit = self._int("blocked_limit", 20)
        if limit <= 0:
            return
        try:
            files = sorted(
                (p for p in self.blocked_dir.iterdir() if p.is_file()),
                key=lambda p: p.stat().st_mtime,
            )
        except OSError:
            return
        for path in files[: max(0, len(files) - limit)]:
            try:
                path.unlink()
            except OSError:
                pass

    def _chat_label(self, event: AstrMessageEvent) -> str:
        group = str(event.get_group_id() or "")
        return f"group:{group}" if group else "private"

    def _log_draw(self, record: guard_mod.DrawRecord) -> None:
        record.user = record.user or "-"
        self._draw_log.write(record)

    def _float(self, key: str, default: float) -> float:
        try:
            return float(self._get(key, default))
        except (TypeError, ValueError):
            return default

    # ------------------------------------------------------------ LLM 工具
    #
    # 自然语言入口。与 `/画图` 命令**共存**：命令稳定可预测，工具负责理解口语。
    # 工具描述里刻意写死「明确要求才调用」，否则群里随便一句话都会被画图刷屏。

    @filter.llm_tool(name="draw_image")
    async def draw_image(
        self,
        event: AstrMessageEvent,
        prompt: str,
        preset: str = "",
        reference: str = "",
        use_reference: bool = False,
        denoise: float = 0.0,
    ) -> str:
        """用本机 ComfyUI（Stable Diffusion / Illustrious）生成或修改图片。

        仅当对方**明确要求**出图时才调用（例如「画个该角色」「来一张猫娘」「用那张立绘换个姿势」）。
        不要主动调用，也不要因为闲聊里出现「画」字就调用。

        关于提示词语言：底层模型只认**英文 danbooru 标签**，中文几乎不起作用。所以
        请把你理解到的需求**自己翻译成英文标签**再填进 prompt，不要原样传中文。
        例：对方说「坐着看书」→ 你填 `sitting, reading book, holding book`。

        关于参考图与姿势：用参考图（图生图）时，**参考图的姿势会被基本锁住**，
        改姿势的提示词几乎无效。所以对方要求换姿势（坐着/躺下/蹲下等）而参考图是
        站姿时，应当**不要**传 reference，让模型自由构图。

        Args:
            prompt(string): **英文 danbooru 标签**。已指定 preset 时只写要追加或改变的部分（`sitting, reading book`），不要重复角色特征；未指定 preset 时才写完整描述（`1girl, solo, white hair, kimono`）。
            preset(string): 预设名（角色或风格）。可用值见返回值里的列表，例如 校服、和服、便服。留空表示不套预设。
            reference(string): 参考图名。留空表示不用参考图。只有对方明确说「用那张立绘」「参考这张」时才填；要求换姿势时不要填。
            use_reference(boolean): 参考图的用法。false=当底图小幅重绘（画风最像官图，适合只换表情/微调）；true=只锁姿势结构、内容重画（适合换服装但保留这个姿势）。
            denoise(number): 重绘幅度 0~1。越小越接近参考图（默认 0.55）；不确定就填 0。
        """
        if not self._bool("enable", True):
            return "绘图功能已关闭。"
        if not self._allowed(event):
            return "当前会话不允许使用绘图功能。"

        # 没有任何具体信息时，只回报可选项，别瞎画
        if not prompt.strip() and not preset.strip() and not reference.strip():
            return self._tool_options()

        # 提示词门与配额检查和 /画图 命令走同一套，不能因为走自然语言就绕过去。
        # 顺序与命令入口一致：内容门 → 参数校验 → 配额，理由见 draw()。
        hits = self._check_prompt_safety(prompt, event)
        if hits:
            self._log_draw(
                guard_mod.DrawRecord(
                    status="prompt_blocked",
                    user=str(event.get_sender_id() or ""),
                    chat=self._chat_label(event),
                    request=prompt,
                    note="via llm_tool / 命中关键词：" + "、".join(hits),
                )
            )
            # 要返回给模型看的一句话（模型再转述给用户），所以这里保留
            # 「不适合发出来」的明确说法，但不念出具体命中的词。
            return "这个描述里有不适合发出来的内容，换个说法或换成正常着装再试。"

        # use_reference 明确决定参考图怎么用，不留给预设的 mode 去猜：
        # true = 只锁姿势结构；false = 当底图小幅重绘
        want_mode = "controlnet" if use_reference else "img2img"
        if use_reference and not reference.strip():
            # 预设自己可能已经绑定了参考图（预设定义里的 ref 字段），
            # 那种情况下不该再要求调用方显式给 reference
            preset_ref = ""
            if preset.strip():
                found = presets_mod.find_preset(self.presets, preset.strip())
                preset_ref = found.ref if found is not None else ""
            if not preset_ref:
                return (
                    "锁姿势模式需要指定参考图：可以用带参考图的预设"
                    "（即预设定义里带了 ref 的那种），或直接填 reference 参数。"
                    + self._tool_options()
                )

        try:
            workflow, chosen, ref, mode = await self._resolve_target(
                "", preset.strip(), prompt.strip(), False, reference.strip(), want_mode
            )
        except ComfyError as exc:
            return f"{exc}"

        # 配额检查放在参数校验之后：名字写错应该报「没有这套预设」，
        # 而不是先甩一句「等 10 秒」把额度也扣掉
        limit = self._check_rate(event)
        if not limit.allowed:
            self._log_draw(
                guard_mod.DrawRecord(
                    status="rate_limited",
                    user=str(event.get_sender_id() or ""),
                    chat=self._chat_label(event),
                    request=prompt,
                    note="via llm_tool / " + limit.reason,
                )
            )
            return f"暂时画不了：{limit.reason}"

        # 优雅降级：模型可能传错 use_reference，或者所选预设绑定的工作流根本没有
        # ControlNet 节点。与其整单失败，不如退到这条工作流实际支持的模式。
        mode = mode or want_mode
        dropped_ref = False
        if ref is not None:
            api = self._converted(workflow)
            if mode == "controlnet" and not uigraph.has_controlnet(api):
                logger.info(
                    f"[comfyui] 工作流「{workflow.label}」没有 ControlNet 节点，"
                    "本次按图生图处理"
                )
                mode = "img2img"
            elif mode != "controlnet" and not uigraph.is_img2img(api):
                mode = "controlnet"

        # 换姿势 / 换场景时参考图会把姿势和背景焊死 —— 即使模型传了 reference 也要摘掉它
        if ref is not None and mode != "controlnet" and _needs_free_composition(prompt):
            why = "换姿势" if _wants_new_pose(prompt) else "换场景"
            logger.info(f"[comfyui] 检测到{why}意图，自动去掉参考图「{ref.label}」")
            ref = None
            dropped_ref = True
            mode = "img2img"
            try:
                workflow = await self._pick_text2img_workflow("", chosen, prompt)
            except ComfyError as exc:
                return f"换姿势时挑工作流失败：{exc}"

        await self._start_generation(
            event,
            workflow,
            prompt.strip(),
            chosen,
            ref,
            denoise if denoise and denoise > 0 else None,
            mode,
            notices=[f"{why}：不用参考图"] if dropped_ref else None,
            exempt=self._nsfw_exempt(event),
        )
        # 实际发送由后台任务完成；这里只回一句极简确认，避免污染模型上下文
        bits = [f"工作流 {workflow.label}"]
        if chosen is not None:
            bits.append(f"预设 {chosen.label}")
        if ref is not None:
            bits.append(
                f"参考图 {ref.label}（{'锁姿势' if mode == 'controlnet' else '当底图'}）"
            )
        elif dropped_ref:
            bits.append("已按换姿势要求去掉参考图")
        return "已开始出图（" + "，".join(bits) + "）。图片会稍后单独发出，不要重复调用本工具。"

    def _tool_options(self) -> str:
        """给模型看的可选项目录。刻意保持简短，避免占满上下文。"""
        presets = "、".join(p.key for p in self.presets.values()) or "（无）"
        refs = "、".join(self.refs.keys()) or "（无）"
        return (
            f"可选预设：{presets}；可选参考图：{refs}。"
            "请用 preset / reference 指定其中之一后重新调用；"
            "参考图配合 use_reference=true 表示只锁姿势。"
        )

    # ------------------------------------------------------------ 工作流管理

    def _seed_dir(self) -> pathlib.Path:
        """插件自带的示例工作流目录。"""
        return pathlib.Path(__file__).resolve().parent / "workflows"

    def _seed_if_empty(self) -> None:
        """首次运行时把自带的示例工作流复制到数据目录。

        只在目标目录里一个工作流都没有时才播种，之后用户的增删不会被覆盖。
        """
        try:
            existing = [
                p
                for p in self.workflow_dir.glob("*.json")
                if p.name not in _NON_WORKFLOW_FILES
            ]
        except OSError:
            return
        if existing:
            return
        seed_dir = self._seed_dir()
        if not seed_dir.is_dir():
            return
        copied = 0
        for src in sorted(seed_dir.glob("*.json")):
            try:
                shutil.copy2(src, self.workflow_dir / src.name)
                copied += 1
            except OSError as exc:
                logger.warning(f"[comfyui] 复制示例工作流 {src.name} 失败：{exc}")
        if copied:
            logger.info(f"[comfyui] 已播种 {copied} 个示例工作流到 {self.workflow_dir}")

    def _meta_path(self) -> pathlib.Path:
        return self.workflow_dir / "workflow_meta.json"

    def _load_meta(self) -> dict[str, Any]:
        path = self._meta_path()
        if not path.exists():
            return {}
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning(f"[comfyui] workflow_meta.json 读取失败：{exc}")
            return {}

    def _load_workflows(self) -> None:
        """扫描工作流目录，建立 key -> Workflow 映射。"""
        self.workflows.clear()
        self._workflow_errors.clear()
        meta = self._load_meta()
        descriptions = meta.get("descriptions") or {}
        slot_map = meta.get("text_slots") or {}

        if not self.workflow_dir.exists():
            return

        for path in sorted(self.workflow_dir.iterdir()):
            if path.suffix.lower() != ".json" or not path.is_file():
                continue
            # 元数据 / 预设文件本身不是工作流
            if path.name in _NON_WORKFLOW_FILES:
                continue
            key = path.stem
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                self._workflow_errors[key] = f"JSON 解析失败：{exc}"
                continue
            if not isinstance(raw, dict):
                self._workflow_errors[key] = "顶层不是 JSON 对象"
                continue

            description = ""
            if isinstance(descriptions, dict):
                description = str(descriptions.get(path.name) or descriptions.get(key) or "")
            slots = []
            if isinstance(slot_map, dict):
                raw_slots = slot_map.get(path.name) or slot_map.get(key) or []
                if isinstance(raw_slots, list):
                    slots = [str(s) for s in raw_slots]

            self.workflows[key] = Workflow(
                key=key,
                path=path,
                description=description,
                text_slots=slots,
                raw=raw,
            )

        # 有专属工作流目录时，也把 ComfyUI 自己的 workflows 目录纳入
        for extra_dir in self._extra_workflow_dirs():
            if not extra_dir.exists():
                continue
            for path in sorted(extra_dir.glob("*.json")):
                if path.name in _NON_WORKFLOW_FILES:
                    continue
                key = path.stem
                if key in self.workflows:
                    continue  # 插件目录优先
                try:
                    raw = json.loads(path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError) as exc:
                    self._workflow_errors[key] = f"JSON 解析失败：{exc}"
                    continue
                if not isinstance(raw, dict):
                    continue
                self.workflows[key] = Workflow(key=key, path=path, raw=raw)

        logger.info(
            f"[comfyui] 载入 {len(self.workflows)} 个工作流"
            + (f"，{len(self._workflow_errors)} 个解析失败" if self._workflow_errors else "")
        )

    def _extra_workflow_dirs(self) -> list[pathlib.Path]:
        raw = str(self._get("extra_workflow_dirs", "") or "")
        dirs: list[pathlib.Path] = []
        for chunk in raw.replace("\n", ";").split(";"):
            chunk = chunk.strip().strip('"')
            if chunk:
                dirs.append(pathlib.Path(chunk))
        return dirs

    def _resolve_workflow(self, key: str) -> Workflow:
        """按 key 精确/模糊找一份工作流。"""
        wanted = (key or "").strip()
        if not wanted:
            raise ComfyError("没指定工作流名")
        if wanted in self.workflows:
            return self.workflows[wanted]
        lowered = wanted.lower()
        exact = [wf for name, wf in self.workflows.items() if name.lower() == lowered]
        if exact:
            return exact[0]
        partial = [wf for name, wf in self.workflows.items() if lowered in name.lower()]
        if len(partial) == 1:
            return partial[0]
        if len(partial) > 1:
            names = "、".join(wf.label for wf in partial)
            raise ComfyError(f"「{wanted}」匹配到多个工作流：{names}")
        raise ComfyError(f"没有叫「{wanted}」的工作流，用「/{self._command()} 工作流」看列表")

    def _pick_workflow(self, requested: str, text: str = "") -> Workflow:
        """决定用哪个工作流。

        优先级：显式 -w 指定 > 关键词路由 > 配置的默认工作流 > 唯一可用项。
        关键词排在默认工作流**之前**，否则「提到某个角色就换 LoRA 工作流」
        这类规则会被默认值吃掉、永远不生效。
        """
        if requested:
            return self._resolve_workflow(requested)

        candidates = [wf for wf in self.workflows.values() if wf.key not in self._workflow_errors]
        if not candidates:
            raise ComfyError(
                f"还没有可用工作流。把 ComfyUI 导出的工作流 JSON 放进 "
                f"{self.workflow_dir} 再试（或用「/{self._command()} 重载」）"
            )

        routed = self._route_by_keyword(text)
        if routed is not None:
            return routed

        default_name = str(self._get("default_workflow", "") or "").strip()
        if default_name:
            try:
                return self._resolve_workflow(default_name)
            except ComfyError:
                logger.warning(
                    f"[comfyui] 配置的默认工作流「{default_name}」不存在，改按关键词/唯一项挑选"
                )

        if len(candidates) == 1:
            return candidates[0]
        names = "、".join(wf.label for wf in candidates)
        raise ComfyError(
            f"有多个工作流可用：{names}\n"
            f"请用「/{self._command()} -w 工作流名 描述」，或在插件配置里设置默认工作流"
        )

    def _route_by_keyword(self, text: str) -> Workflow | None:
        """按配置的关键词表挑工作流；没有命中返回 None。

        支持两种写法::

            my_lora_workflow: mychar, 我的角色    # 推荐
            我的角色=mylora_workflow               # 兼容

        第一段（`:`` 或 ``->`` 或第一个 ``=`` 之前）只要**是**某个已知工作流名，
        就当成「工作流名 在前」；关键词再按逗号切。这里的逗号切分只作用于
        关键词，所以工作流名里带空格（如 ``illustrious lora``）也不会被切坏。
        """
        if not text:
            return None
        rules = self._get("workflow_rules", "") or ""
        if isinstance(rules, list):
            rules = "\n".join(str(r) for r in rules)
        lowered_text = text.lower()
        for line in str(rules).splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            target, keyword_blob = _split_rule(line)
            if target not in self.workflows:
                continue
            for keyword in keyword_blob:
                if keyword and keyword.lower() in lowered_text:
                    return self.workflows[target]
        return None

    # ------------------------------------------------------------ 参考图

    def _load_refs(self) -> None:
        self.refs = references.load_refs(self.refs_dir)
        notes = references.load_notes(self.refs_dir)
        for key, note in notes.items():
            if key in self.refs:
                self.refs[key].note = note
        if self.refs:
            logger.info(
                f"[comfyui] 载入 {len(self.refs)} 张参考图："
                + "、".join(r.label for r in self.refs.values())
            )

    def _seed_refs_if_empty(self) -> None:
        """首次运行把 refs/ 和说明模板建好（参考图体积大，不随插件分发）。"""
        self.refs_dir.mkdir(parents=True, exist_ok=True)
        meta = self.refs_dir / "refs.json"
        if meta.exists():
            return
        seed = self._seed_dir() / "refs.json"
        if seed.is_file():
            try:
                shutil.copy2(seed, meta)
            except OSError as exc:
                logger.warning(f"[comfyui] 播种 refs.json 失败：{exc}")

    # ------------------------------------------------------------ 预设管理

    def _presets_path(self) -> pathlib.Path:
        custom = str(self._get("presets_file", "") or "").strip().strip('"')
        if custom:
            return pathlib.Path(custom)
        return self.data_dir / "presets.json"

    def _preset_base_dir(self) -> pathlib.Path:
        """预设里相对路径的解析基准（提示词文件通常和预设放一起）。"""
        return self._presets_path().parent

    def _seed_presets_if_empty(self) -> None:
        """首次运行播种一份预设模板（引用你已有的提示词文件）。"""
        target = self._presets_path()
        if target.exists():
            return
        seed = self._seed_dir() / "presets.json"
        if not seed.is_file():
            return
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(seed, target)
            logger.info(f"[comfyui] 已播种预设模板到 {target}")
        except OSError as exc:
            logger.warning(f"[comfyui] 播种预设模板失败：{exc}")

    def _load_presets(self) -> None:
        self._seed_presets_if_empty()
        self.presets, problem = presets_mod.load_presets(self._presets_path())
        if problem:
            logger.warning(f"[comfyui] {problem}")
        if self.presets:
            logger.info(
                f"[comfyui] 载入 {len(self.presets)} 个预设："
                + "、".join(p.label for p in self.presets.values())
            )

    def _resolve_preset(self, wanted: str) -> presets_mod.Preset:
        preset = presets_mod.find_preset(self.presets, wanted)
        if preset is None:
            raise ComfyError(
                f"没有叫「{wanted}」的预设，用「/{self._command()} 预设」看列表"
            )
        return preset

    def _pick_preset(self, requested: str, text: str) -> presets_mod.Preset | None:
        """挑预设：显式 -p > 关键词/别名命中 > 配置的默认预设 > 不套预设。"""
        if requested:
            return self._resolve_preset(requested)
        if not self.presets:
            return None
        hit = presets_mod.find_by_trigger(self.presets, text)
        if hit is not None:
            return hit
        default_name = str(self._get("default_preset", "") or "").strip()
        if default_name:
            found = presets_mod.find_preset(self.presets, default_name)
            if found is not None:
                return found
            logger.warning(f"[comfyui] 配置的默认预设「{default_name}」不存在，本次不套预设")
        return None

    def _preset_base_prompt(self, preset: presets_mod.Preset, workflow: Workflow) -> str:
        """取预设在某工作流下要用的基础提示词。"""
        base = presets_mod.resolve_prompt(preset, workflow.key, self._preset_base_dir())
        if preset.load_error:
            logger.warning(f"[comfyui] 预设「{preset.label}」{preset.load_error}")
        return base

    def _apply_preset_overrides(
        self, prompt: dict[str, Any], preset: presets_mod.Preset, workflow: Workflow
    ) -> list[str]:
        """把预设里的参数覆盖到 prompt 上（尺寸/采样参数/LoRA）。返回说明文字。"""
        notes: list[str] = []

        # 尺寸：预设 > 全局配置
        width = preset.width or self._int("width", 0)
        height = preset.height or self._int("height", 0)
        if width > 0 and height > 0 and uigraph.set_latent_size(prompt, width, height):
            notes.append(f"{width}x{height}")

        sampler_overrides = {
            "steps": preset.steps or None,
            "cfg": preset.cfg or None,
            "sampler_name": preset.sampler_name or None,
            "scheduler": preset.scheduler or None,
        }
        applied = uigraph.set_sampler_params(prompt, **sampler_overrides)
        if applied:
            notes.append("参数按预设")

        if preset.lora:
            if uigraph.set_lora(prompt, preset.lora):
                notes.append(f"LoRA {preset.lora}")
            else:
                # ⚠ 前缀 = 要提醒用户的问题。不带前缀的是「参数已生效」这类流水账，
                # 只进日志；短回执靠这个前缀筛出真正该说的话。
                notes.append(f"⚠ 工作流没有 LoRA 节点，未应用 {preset.lora}")
        return notes

    # ------------------------------------------------------------ ComfyUI 交互

    def _client_or_create(self) -> ComfyClient:
        server = self._server()
        if self._client is None or self._client.server != ComfyClient._normalize(server):
            self._client = ComfyClient(server, timeout=self._int("timeout", 300))
            self._object_info = None
        return self._client

    async def _object_info_cached(self, refresh: bool = False) -> dict[str, Any]:
        if self._object_info is None or refresh:
            client = self._client_or_create()
            self._object_info = await client.object_info(refresh=refresh)
        return self._object_info

    def _converted(self, workflow: Workflow) -> dict[str, Any]:
        """取工作流转换后的 API prompt（同步版，用于「能不能图生图」这类判断）。

        只在 ``_object_info`` 已经缓存后可用；没缓存时抛 ComfyError，
        由调用方决定是否先 await 一次 ``_object_info_cached()``。
        """
        if workflow._api is None:
            if self._object_info is None:
                raise ComfyError(f"节点定义尚未加载，无法判断「{workflow.label}」")
            workflow._api = uigraph.convert(workflow.raw, self._object_info)
        return workflow._api

    async def _build_prompt_for(
        self,
        workflow: Workflow,
        user_text: str,
        preset: presets_mod.Preset | None,
        reference: references.Reference | None = None,
        denoise_override: float | None = None,
        mode: str = "img2img",
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """带预设地组装 prompt。"""
        object_info = await self._object_info_cached()
        if workflow._api is None:
            workflow._api = uigraph.convert(workflow.raw, object_info)
        # 每次生成都从副本出发，避免上一轮的提示词、种子残留
        prompt = json.loads(json.dumps(workflow._api))

        # 预设名本身不是提示词，留着只会变成正向提示词里的中文噪声 tag
        if preset is not None:
            user_text, used_trigger = presets_mod.strip_trigger(preset, user_text)
            if used_trigger:
                logger.debug(f"[comfyui] 从描述里摘掉预设触发词「{used_trigger}」")

        # 中文需求先翻成英文 danbooru 标签：底层模型不认中文，直接送进去等于没写。
        # 这一步必须在下面的换姿势判断之前做，翻译出来的 `sitting` 也要能被识别。
        user_text, translated = _translate_cn_phrases(user_text or "")
        if translated:
            logger.info(
                "[comfyui] 中文提示词已翻译："
                + "；".join(f"{cn} → {tags}" for cn, tags in translated)
            )
        leftover_cn = _leftover_cjk(user_text)
        if leftover_cn:
            logger.warning(
                f"[comfyui] 提示词里还有没翻出来的中文：{leftover_cn}"
                "（底层模型只认英文 danbooru 标签，这些词多半无效甚至干扰画面）"
            )

        # 模型预检：缺模型时给出人话，而不是 ComfyUI 的 value_not_in_list
        missing = uigraph.missing_models(prompt, object_info)
        if missing:
            head = "；".join(missing[:4])
            more = f"（还有 {len(missing) - 4} 项）" if len(missing) > 4 else ""
            raise ComfyError(
                f"工作流引用的模型本机没有：{head}{more}。"
                "请把模型装到 ComfyUI 的对应目录，或改用别的工作流（/画图 -w 工作流名）。"
            )

        preset_base = ""
        preset_tags = ""
        preset_notes: list[str] = []
        if preset is not None:
            preset_base = self._preset_base_prompt(preset, workflow)
            preset_tags = preset.quality_tags
            if not preset_base:
                logger.warning(
                    f"[comfyui] 预设「{preset.label}」没解析出基础提示词，本次只用用户输入"
                )

        # 用户要换姿势时，预设里那句字面量 `standing` 会把姿势钉死（中文提示词
        # 权重太低，压不过它）。必须把它清掉，否则「自动去参考图」白做。
        if _wants_new_pose(user_text) and preset_base:
            stripped_base, removed = _strip_pose_tags(preset_base)
            if removed:
                logger.info(
                    f"[comfyui] 换姿势请求：从预设「{preset.label}」基础提示词里"
                    f"摘掉姿势 tag {removed}"
                )
                preset_notes.append("已去掉预设里固定的姿势：" + "、".join(removed))
                preset_base = stripped_base

        positive = self._compose_prompt(user_text, preset_base, preset_tags)
        negative = (
            (preset.negative_prompt if preset and preset.negative_prompt else "")
            or str(self._get("negative_prompt", "") or "")
        ).strip() or None
        uigraph.apply_prompt(prompt, positive, negative=negative)

        if preset is not None:
            preset_notes = self._apply_preset_overrides(prompt, preset, workflow)

        # 参考图：转白底 → 缩放 → 上传 → 写进取图节点 → 设 denoise
        ref_note = ""
        if reference is not None:
            ref_note = await self._apply_reference(
                prompt, reference, preset, denoise_override, mode
            )
        seed = self._int("seed", -1)
        uigraph.randomize_seeds(prompt, seed if seed >= 0 else None)

        summary = uigraph.describe(prompt)
        if summary.get("size") is None and reference is not None:
            # 图生图的画布来自取图节点（没有 EmptyLatentImage 的 width/height），
            # describe 抽不到尺寸。它其实就是参考图缩放后的尺寸 —— 单独算一下，
            # 否则日志里 size 永远是空的，事后查「这张多大」就查不到。
            try:
                summary["size"] = list(
                    references.prepared_size(
                        reference.path, max_side=self._int("ref_max_side", 1536)
                    )
                )
            except Exception:  # noqa: BLE001 - 算不出来就算了，不影响出图
                pass
        summary["preset"] = preset.label if preset is not None else ""
        summary["preset_notes"] = preset_notes
        summary["positive_len"] = len(positive)
        summary["reference"] = reference.label if reference is not None else ""
        summary["reference_note"] = ref_note
        summary["translated"] = translated
        summary["leftover_cn"] = leftover_cn
        return prompt, summary

    async def _apply_reference(
        self,
        prompt: dict[str, Any],
        reference: references.Reference,
        preset: presets_mod.Preset | None,
        denoise_override: float | None,
        mode: str = "img2img",
    ) -> str:
        """把参考图写进 prompt。返回一句用于回执的说明。"""
        if not uigraph.is_img2img(prompt):
            raise ComfyError(
                f"「{reference.label}」是参考图，但当前工作流没有取图节点（LoadImage）。"
                "请换一份带取图节点的工作流。"
            )

        max_side = self._int("ref_max_side", 1536)
        try:
            blob = references.prepare_image(reference.path, max_side=max_side)
        except references.RefError as exc:
            raise ComfyError(f"处理参考图失败：{exc}") from exc

        name = references.upload_name(reference.path, blob)
        client = self._client_or_create()
        uploaded = await client.upload_image(name, blob)

        if not uigraph.set_input_image(prompt, uploaded):
            raise ComfyError(
                f"工作流里没有能写入的取图输入，参考图「{reference.label}」未能生效"
            )

        if mode == "controlnet":
            return self._apply_controlnet(prompt, reference, preset, uploaded, max_side)

        # denoise 优先级：命令行 -d > 预设 > 全局配置
        denoise = denoise_override
        if denoise is None and preset is not None and preset.denoise > 0:
            denoise = preset.denoise
        if denoise is None:
            denoise = float(self._int("img2img_denoise", 55)) / 100.0
        denoise = max(0.0, min(1.0, float(denoise)))
        if not uigraph.set_denoise(prompt, denoise):
            logger.warning("[comfyui] 工作流里没有采样器，denoise 未设置")

        logger.info(
            f"[comfyui] 参考图 {reference.key} -> {uploaded}，denoise={denoise:.2f}"
        )
        return f"参考图 {reference.label}，重绘 {denoise:.0%}"

    def _apply_controlnet(
        self,
        prompt: dict[str, Any],
        reference: references.Reference,
        preset: presets_mod.Preset | None,
        uploaded: str,
        max_side: int = 1536,
    ) -> str:
        """ControlNet 模式的参数设置。

        与图生图的关键区别：**不降 denoise**。ControlNet 的思路是「结构照抄、
        内容重画」，所以保持高 denoise（默认 1.0）让提示词自由发挥，
        约束力来自 ControlNet 权重而不是底图。
        """
        if not uigraph.has_controlnet(prompt):
            raise ComfyError(
                f"工作流「{reference.label}」没有 ControlNet 节点，无法锁结构。"
                "请换一份带 ControlNetApply 的工作流，或去掉 -c 按图生图跑。"
            )

        # 画布比例跟着参考图走：比例不一致时 ControlNet 会把控制图拉伸，
        # 导致构图整体偏移（实测头部被裁掉）。所以先按参考图比例算画布，
        # 再把控制图缩放到同一尺寸。
        ref_w, ref_h = references.prepared_size(reference.path, max_side)
        width = self._int("width", 0)
        height = self._int("height", 0)
        if width > 0 and height > 0:
            canvas = (width, height)  # 用户显式指定尺寸则尊重它
        else:
            canvas = uigraph.fit_canvas(ref_w, ref_h)
        uigraph.set_latent_size(prompt, *canvas)
        uigraph.set_image_scale_size(prompt, *canvas)

        # union 模式：预设的 control_type 优先，否则用配置默认（auto）
        control_type = (preset.control_type if preset is not None else "") or str(
            self._get("controlnet_type", "auto") or "auto"
        )
        type_applied = uigraph.set_control_type(prompt, control_type)

        # 模式与预处理器不匹配会得到不可预期的结果，要明确警告而不是默默跑
        mismatch = uigraph.check_controlnet_mode(prompt, control_type)
        if mismatch:
            logger.warning(f"[comfyui] {mismatch}")

        # 权重：预设 > 全局配置
        strength = preset.control_strength if preset is not None and preset.control_strength > 0 else 0.0
        if strength <= 0:
            strength = float(self._int("controlnet_strength", 80)) / 100.0
        strength = max(0.0, min(2.0, float(strength)))
        uigraph.set_control_strength(prompt, strength)

        # 预设给了 denoise 就尊重它（想同时用底图时可以调低），否则保持工作流原值
        notes: list[str] = []
        if preset is not None and preset.denoise > 0:
            uigraph.set_denoise(prompt, preset.denoise)
            notes.append(f"重绘 {preset.denoise:.0%}")
        notes.append(f"{canvas[0]}x{canvas[1]}")

        info = uigraph.describe_controlnet(prompt)
        model = info["models"][0] if info["models"] else "?"
        pre = info["preprocessors"][0] if info["preprocessors"] else "?"
        logger.info(
            f"[comfyui] ControlNet {reference.key} -> {uploaded}，"
            f"model={model} type={control_type if type_applied else '(模型自带)'} "
            f"strength={strength:.2f} pre={pre} 画布={canvas[0]}x{canvas[1]}"
        )
        type_note = f"模式 {control_type}" if type_applied else "模型自带模式"
        warn_note = "（模式与预处理器不匹配，建议看日志）" if mismatch else ""
        return (
            f"参考图 {reference.label} 锁结构（{type_note}，权重 {strength:.0%}）"
            + ("，" + " / ".join(notes) if notes else "")
            + warn_note
        )

    def _compose_prompt(
        self,
        user_text: str,
        preset_base: str = "",
        preset_tags: str = "",
    ) -> str:
        """拼出最终的正向提示词。

        顺序很关键：**预设基础提示词必须排在最前面**。角色 LoRA 要求触发词
        （例如 ``my_char``，就是你 LoRA 训练时用的那个触发词）位于正向提示词开头，把画质词或用户输入放到
        它前面会削弱甚至破坏角色还原。

        于是默认模板是 ``{base}, {tags}, {prompt}``：预设 → 画质词 → 用户输入。
        """
        base = (preset_base or "").strip()
        # 预设自带的画质词优先于全局配置，避免和预设里已有的画质词重复叠加
        tags = (preset_tags or "").strip() or str(self._get("quality_tags", "") or "").strip()
        template = str(self._get("prompt_template", "") or "").strip()
        text = (user_text or "").strip()

        if template:
            # ⚠️ 兼容旧配置：`{base}` 是后加的占位符，早期默认模板是
            # `{tags}, {prompt}`。用户的配置是 AstrBot 按当时的默认值生成的，
            # 不会因为插件升级而更新，于是模板里没有 `{base}` —— 预设里那几百
            # 字符的角色提示词（含 LoRA 触发词）会被**静默丢弃**，出图直接崩成
            # 随机角色。所以这里检测到模板缺 `{base}` 就自动补在最前面。
            if _PLACEHOLDER_BASE not in template and base:
                template = f"{_PLACEHOLDER_BASE}, {template}"
            rendered = (
                template.replace(_PLACEHOLDER_BASE, base)
                .replace(_PLACEHOLDER_TAGS, tags)
                .replace(_PLACEHOLDER_PROMPT, text)
            )
        else:
            rendered = ", ".join(part for part in (base, tags, text) if part)

        # 去重：预设自带的画质词和全局 quality_tags 往往重合（预设是从工作流
        # 里抄的、全局是另一份），拼起来会出现两遍 "masterpiece, best quality…"。
        # 重复 tag 会白白占权重，所以按忽略大小写去重，保留首次出现的位置。
        parts = [part.strip() for part in rendered.split(",")]
        seen: set[str] = set()
        unique: list[str] = []
        for part in parts:
            if not part:
                continue
            marker = part.lower()
            if marker in seen:
                continue
            seen.add(marker)
            unique.append(part)
        return ", ".join(unique)

    async def _check_server(self) -> str:
        """返回一句服务端摘要，顺便验证连通性。"""
        client = self._client_or_create()
        stats = await client.system_stats()
        device = (stats.get("devices") or [{}])[0]
        name = device.get("name") or "unknown"
        try:
            vram = round(int(device.get("vram_total") or 0) / 1024**3, 1)
        except (TypeError, ValueError):
            vram = 0
        queue = await client.queue_state()
        return f"{name} / {vram}GB 显存 / 队列 {queue['running']} 运行中 + {queue['pending']} 等待"

    # ------------------------------------------------------------ 命令处理

    @filter.command("画图", alias={"文生图", "comfyui"})
    async def draw(self, event: AstrMessageEvent):
        """画图 / 工作流管理。用法：/画图 <描述> | /画图 -w <工作流> <描述> | /画图 工作流"""
        if not self._allowed(event):
            return
        raw = (event.message_str or "").strip()
        command = self._command()

        # 子命令
        normalized = raw.replace("　", " ").strip()
        for prefix in (command, "画图", "文生图", "comfyui"):
            if normalized.startswith(prefix):
                normalized = normalized[len(prefix) :].strip()
                break

        if normalized in ("工作流", "列表", "list", "-l", "--list"):
            async for item in self._list_workflows(event):
                yield item
            return
        if normalized.startswith("信息") or normalized in ("详情", "info"):
            name = normalized[2:].strip() if normalized.startswith("信息") else ""
            async for item in self._workflow_info(event, name):
                yield item
            return
        if normalized in ("重载", "reload"):
            self._load_workflows()
            self._load_presets()
            self._load_refs()
            self._load_persona()
            self._object_info = None
            yield event.plain_result(
                f"已重新扫描：{len(self.workflows)} 个工作流、"
                f"{len(self.presets)} 个预设、{len(self.refs)} 张参考图、"
                f"{sum(len(v) for v in self._persona.values())} 句台词"
                + (
                    f"，{len(self._workflow_errors)} 个工作流有问题"
                    if self._workflow_errors
                    else ""
                )
            )
            return
        if normalized in ("人设", "台词", "persona"):
            yield event.plain_result(
                persona_mod.render_persona(self._persona, self.persona_path)
            )
            return
        if normalized in ("参考图", "立绘", "refs", "-r"):
            async for item in self._list_refs(event):
                yield item
            return
        if normalized in ("预设", "角色", "presets", "-p"):
            async for item in self._list_presets(event):
                yield item
            return
        if normalized.startswith("预设信息") or normalized.startswith("角色信息"):
            name = normalized[4:].strip()
            async for item in self._preset_info(event, name):
                yield item
            return
        if normalized in ("状态", "status", "ping"):
            try:
                report = await self._check_server()
            except ComfyError as exc:
                yield event.plain_result(f"ComfyUI 连不上：{exc}")
                return
            yield event.plain_result(f"ComfyUI 正常：{report}")
            return
        if normalized.startswith(("日志", "log")):
            count_text = normalized[2:].strip()
            count = 10
            if count_text.isdigit():
                count = max(1, min(50, int(count_text)))
            lines = self._draw_log.tail(count)
            if not lines:
                yield event.plain_result(
                    f"还没有出图记录。日志会写在：\n{self._draw_log.path}"
                )
                return
            yield event.plain_result(
                f"最近 {len(lines)} 条出图记录（完整日志：{self._draw_log.path}）：\n"
                + "\n".join(lines)
            )
            return

        # 出图：先解析 -w / -p / -r / -c / -d / -n 开关
        (
            requested,
            preset_name,
            skip_preset,
            reference_name,
            text,
            denoise,
            ref_mode,
        ) = _parse_draw_args(normalized)
        # 只给参考图、不写提示词也允许（纯「重画这张」）
        if not text and not reference_name:
            yield event.plain_result(_usage(command))
            return

        # 提示词里的露骨内容：生成之前拦掉，省得白跑一张图。
        # 放在配额检查之前 —— 内容问题和配额是两回事，不该因为「这词不能用」
        # 反而消耗掉一次出图额度。
        hits = self._check_prompt_safety(text, event)
        if hits:
            self._log_draw(
                guard_mod.DrawRecord(
                    status="prompt_blocked",
                    user=str(event.get_sender_id() or ""),
                    chat=self._chat_label(event),
                    request=text,
                    note="命中关键词：" + "、".join(hits),
                )
            )
            # 台词故意含糊：具体命中了什么词只写日志。在群里把词念出来，
            # 等于替请求者向所有人广播他刚才想画什么。
            await self._speak(
                event,
                "denied_prompt",
                fallback="这个描述里有不适合发出来的内容，换一个吧～",
            )
            return

        try:
            workflow, preset, reference, mode = await self._resolve_target(
                requested, preset_name, text, skip_preset, reference_name, ref_mode
            )
        except ComfyError as exc:
            yield event.plain_result(str(exc))
            return

        # 频率限制放在参数校验**之后**：上面那些是「名字写错了」这类问题，
        # 应该报出有用的错误，而不是先甩一句「等 10 秒」。
        # 又放在真正出图**之前**：解析工作流/预设/参考图只是查表，没有网络和显卡开销，
        # 所以被限流时依然没有白活儿。
        limit = self._check_rate(event)
        if not limit.allowed:
            self._log_draw(
                guard_mod.DrawRecord(
                    status="rate_limited",
                    user=str(event.get_sender_id() or ""),
                    chat=self._chat_label(event),
                    request=text or reference_name,
                    note=limit.reason,
                )
            )
            # 台词 + 具体要等多久。等多久是用户必须知道的信息，
            # 不能因为要有个性就把它吞掉，所以台词后面接一句实情。
            line = self._line("denied_rate")
            await self._safe_send(
                event, f"{line}（{limit.reason}）" if line else limit.reason
            )
            return

        # 要求换姿势时自动去掉参考图：参考图会把姿势焊死，留着它提示词等于白写。
        # 用户显式写了 -r 就尊重他的选择（他自己要拿那张图当底）。
        notices: list[str] = []
        if reference is not None and not reference_name and _needs_free_composition(text):
            why = "换姿势" if _wants_new_pose(text) else "换场景"
            logger.info(
                f"[comfyui] 检测到{why}意图（{text[:30]}），自动去掉参考图"
                f"「{reference.label}」——参考图会锁住姿势和背景"
            )
            notices.append(f"{why}：不用参考图")
            dropped = reference.label
            reference = None
            mode = "img2img"
            # 参考图没了，必须换回文生图工作流：img2img 工作流的尺寸来自取图节点，
            # 缺了参考图会退化成 768×768 并出崩坏的图。
            try:
                workflow = await self._pick_text2img_workflow(requested, preset, text)
            except ComfyError as exc:
                yield event.plain_result(f"换成不用参考图时挑工作流失败：{exc}（原参考图 {dropped}）")
                return

        await self._start_generation(
            event,
            workflow,
            text,
            preset,
            reference,
            denoise,
            mode or "img2img",
            notices=notices,
        )

    async def _resolve_target(
        self,
        requested_workflow: str,
        preset_name: str,
        text: str,
        skip_preset: bool,
        reference_name: str = "",
        ref_mode: str = "",
    ) -> tuple[Workflow, presets_mod.Preset | None, references.Reference | None, str]:
        """把一次出图请求解析成 ``(工作流, 预设, 参考图)``。

        优先级：``-w`` 显式指定 > 预设指定的工作流 > 关键词路由 > 默认工作流。
        给了参考图时会优先选一份能做图生图的工作流（含取图节点）。

        预设知道自己的角色该配哪个工作流（例如带角色 LoRA 的那份），所以
        **预设指定的工作流必须优先于全局默认工作流**；否则会拿一个没有 LoRA
        节点的工作流去跑，LoRA 静默不生效、角色还原不出来。

        抽成一个方法是因为这段顺序很容易写漏，命令处理器和测试都要走同一条路。
        """
        preset: presets_mod.Preset | None = None
        if not skip_preset:
            preset = self._pick_preset(preset_name, text)

        reference: references.Reference | None = None
        if reference_name:
            reference = references.find_reference(self.refs, reference_name)
            if reference is None:
                raise ComfyError(
                    f"没有叫「{reference_name}」的参考图，"
                    f"用「/{self._command()} 参考图」看列表"
                )
        elif preset is not None and preset.ref:
            reference = references.find_reference(self.refs, preset.ref)
            if reference is None:
                logger.warning(
                    f"[comfyui] 预设「{preset.label}」指定的参考图 "
                    f"「{preset.ref}」不存在，本次按文生图处理"
                )

        if reference is not None:
            # 参考图怎么用：命令行 -c > 预设 mode > 全局默认
            mode = ref_mode or (preset.mode if preset is not None else "")
            mode = (mode or str(self._get("ref_mode", "") or "")).strip().lower()
            if mode not in ("controlnet", "img2img"):
                mode = "img2img"
            workflow = await self._pick_reference_workflow(
                requested_workflow, preset, mode
            )
        else:
            mode = ""
            workflow = self._pick_workflow(requested_workflow, text)
            if preset is not None and not requested_workflow and preset.workflow:
                try:
                    workflow = self._resolve_workflow(preset.workflow)
                except ComfyError:
                    logger.warning(
                        f"[comfyui] 预设「{preset.label}」指定的工作流 "
                        f"「{preset.workflow}」不存在，沿用 {workflow.label}"
                    )
        return workflow, preset, reference, mode

    async def _pick_reference_workflow(
        self, requested: str, preset: presets_mod.Preset | None, mode: str
    ) -> Workflow:
        """挑一份能用参考图的工作流。

        ``mode`` 决定要求：

        * ``controlnet`` —— 必须有 ControlNet 应用节点（参考图只提供结构/姿势）
        * ``img2img``   —— 必须有取图节点（参考图当底图重绘）

        两种情况都要明确报错而不是悄悄退化成别的模式，否则用户会以为
        参考图生效了、其实完全没起作用。
        """
        # 判断节点构成需要节点定义表，这里先确保它加载好
        await self._object_info_cached()
        want_controlnet = mode == "controlnet"
        candidates: list[Workflow] = []
        #: 预设显式绑定的工作流属于「作者指定」，可以放宽模式匹配：
        #: 预设作者知道自己的角色该配哪份工作流，模式不匹配交给上层降级处理。
        preset_bound = False

        if requested:
            candidates = [self._resolve_workflow(requested)]
        else:
            key = "default_controlnet_workflow" if want_controlnet else "default_img2img_workflow"
            configured = str(self._get(key, "") or "").strip()
            if configured:
                try:
                    candidates = [self._resolve_workflow(configured)]
                except ComfyError:
                    logger.warning(f"[comfyui] 配置的工作流「{configured}」不存在")
            if not candidates and preset is not None and preset.workflow:
                try:
                    candidates = [self._resolve_workflow(preset.workflow)]
                    preset_bound = True
                except ComfyError:
                    pass
            if not candidates:
                # 自动找：先按名字，再逐个用节点构成判断
                named = [
                    w
                    for w in self.workflows.values()
                    if ("controlnet" if want_controlnet else "img2img") in w.key.lower()
                ]
                candidates = named or list(self.workflows.values())

        def usable(workflow: Workflow) -> bool:
            try:
                api = self._converted(workflow)
            except Exception:  # noqa: BLE001 - 转换失败的工作流直接跳过
                return False
            if not uigraph.is_img2img(api):
                return False  # 没有取图节点，参考图无处注入
            if want_controlnet and not preset_bound:
                return uigraph.has_controlnet(api)
            return True

        for workflow in candidates:
            if usable(workflow):
                return workflow

        names = "、".join(w.label for w in self.workflows.values())
        need = "ControlNet 应用节点 + 取图节点" if want_controlnet else "取图节点（LoadImage）"
        raise ComfyError(
            f"没有可用于{'锁结构' if want_controlnet else '图生图'}的工作流（需要{need}）。\n"
            f"现有工作流：{names}\n"
            "把对应工作流 JSON 放进工作流目录，或在配置里指定默认工作流。"
        )

    async def _start_generation(
        self,
        event: AstrMessageEvent,
        workflow: Workflow,
        text: str,
        preset: presets_mod.Preset | None = None,
        reference: references.Reference | None = None,
        denoise: float | None = None,
        mode: str = "img2img",
        notices: list[str] | None = None,
        exempt: bool = False,
    ) -> None:
        """立刻回执，然后把生成丢到后台任务里。

        回执默认**只说一句**（``verbose_replies`` 打开才说详细内容）：
        工作流、预设、参考图、参数、耗时、暴露度全部写进 ``draw.log``，
        聊天里不再刷屏。用户明确要求过这一点。
        """
        umo = event.unified_msg_origin
        lock = self._session_locks.setdefault(umo, asyncio.Lock())
        if lock.locked():
            await event.send(
                event.plain_result("这个会话已经有一张在画了，等它出来～")
            )
            return

        try:
            report = await self._check_server()
        except ComfyError as exc:
            await event.send(
                event.plain_result(
                    f"连不上 ComfyUI：{exc}\n请确认 ComfyUI 已启动，"
                    "且插件配置里的地址正确。"
                )
            )
            return

        queue_note = ""
        if "队列" in report and "0 运行中 + 0 等待" not in report:
            queue_note = f"（排队：{report.split('/')[-1].strip()}）"

        # ---- 台词 + 内容判定，一次算清 ----
        #
        # 「这次会话能不能收露骨产出」只在这里算**一次**，然后一路带到底
        # （开始台词、产出图门都用同一个值）。以前产出图门里又自己调了一次
        # `_nsfw_exempt()`，两处独立求值在正常情况下结果一样，但一旦中途
        # 配置/名单有变就会不一致 —— 而「开始说可以、出图又说不行」是最难查的。
        exempt = self._nsfw_exempt(event)
        if notices:
            situation = "pose" if any("姿势" in n for n in notices) else "scene"
        elif exempt:
            # 私聊 + 白名单：单独一套台词，让用户知道这次是被允许的
            situation = "private_r18"
        else:
            situation = "start"
        if queue_note:
            situation = "queued"
            queue_note = ""  # 台词里已经说了在排队，别再说一遍

        logger.info(
            f"[comfyui] 内容判定：exempt={exempt}（私聊+白名单才能收露骨产出）"
            f" 场景={situation}"
        )

        if self._bool("verbose_replies", False):
            preset_note = f" + 预设「{preset.label}」" if preset is not None else ""
            if reference is not None:
                ref_note = (
                    f" + 锁结构「{reference.label}」"
                    if mode == "controlnet"
                    else f" + 参考图「{reference.label}」"
                    + (f"（重绘 {denoise:.0%}）" if denoise is not None else "")
                )
            else:
                ref_note = ""
            tag = "".join(f"（{n}）" for n in (notices or []))
            ack = (
                f"开始画了，用「{workflow.label}」{preset_note}{ref_note}"
                f"{tag}，出图后发给你。"
            )
            await event.send(event.plain_result(ack))
        else:
            line = self._line(situation)
            # 台词 + 功能提示：queue_note 是「还得等多久」这类用户必须知道的信息，
            # 不能因为要有个性就吞掉；但已经用 queued 台词说过的就不再重复。
            suffix = f"（{queue_note.strip('（）')}）" if queue_note else ""
            await event.send(event.plain_result((line or "🎨 开始画了～") + suffix))

        task = asyncio.create_task(
            self._run_generation(
                event, workflow, text, umo, lock, preset, reference, denoise, mode,
                exempt=exempt,
            ),
            name=f"comfyui-draw-{umo}",
        )
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def _run_generation(
        self,
        event: AstrMessageEvent,
        workflow: Workflow,
        text: str,
        umo: str,
        lock: asyncio.Lock,
        preset: presets_mod.Preset | None = None,
        reference: references.Reference | None = None,
        denoise: float | None = None,
        mode: str = "img2img",
        exempt: bool = False,
    ) -> None:
        """后台执行一次生成，并把结果发回会话。

        ``exempt`` 由调用方在**开始出图前**算好并传进来（不在这里重算）：
        开始台词和产出图门必须用同一个判定，否则会出现「开始说可以、出图又说不行」。
        """
        async with lock:
            async with self._semaphore:
                started = time.monotonic()
                try:
                    prompt, summary = await self._build_prompt_for(
                        workflow, text, preset, reference, denoise, mode
                    )
                except (ComfyError, uigraph.ConversionError) as exc:
                    await self._safe_send(event, f"准备工作流失败：{exc}")
                    return

                if self._bool("show_progress", False):
                    await self._safe_send(event, "已提交给 ComfyUI…")

                client = self._client_or_create()

                # 写盘预检：SaveImage 是最后一步，写不进去会白等一分钟
                if self._bool("check_output_writable", True):
                    problem = await client.check_image_write()
                    if problem:
                        await self._safe_send(
                            event,
                            f"❌ ComfyUI 写不了图片，先不跑了（省得白等）：{problem}\n"
                            "常见原因：\n"
                            "· ComfyUI 没有写入自己 output 目录的权限\n"
                            "· 磁盘满了（尤其 C 盘）\n"
                            "· 从受限环境/沙箱里启动了 ComfyUI\n"
                            "确认后重试即可。",
                        )
                        return

                try:
                    result = await client.generate(prompt)
                except ComfyError as exc:
                    await self._safe_send(event, f"生成失败：{exc}")
                    return
                except asyncio.CancelledError:
                    raise
                except Exception as exc:  # 兜底，别让异常吞掉任务
                    logger.exception("[comfyui] 生成时发生未预期异常")
                    await self._safe_send(event, f"生成出错：{exc}")
                    return

                if result.empty:
                    await self._safe_send(
                        event,
                        "ComfyUI 跑完了，但没有产出图片。常见原因：工作流里的节点把产物"
                        "丢弃了、模型文件缺失，或 SaveImage 写盘失败（磁盘满了 / 输出目录"
                        "没有写权限）。详细报错见 AstrBot 日志里的 [comfyui] 记录。",
                    )
                    return

                sent = 0
                blocked: list[tuple[str, safety_mod.ImageVerdict]] = []
                exposures: list[str] = []
                outputs: list[str] = []
                for item in result.images:
                    suffix = pathlib.Path(item.get("filename", "")).suffix.lower()
                    if suffix and suffix not in _IMAGE_SUFFIXES:
                        continue
                    try:
                        blob = await client.download(item)
                    except ComfyError as exc:
                        logger.warning(f"[comfyui] 下载图片失败：{exc}")
                        continue
                    path = self._save_cache(item, blob)
                    if path is None:
                        continue

                    # 内容兜底：发出去之前检查，群聊一律不放行露骨产出
                    verdict = self._inspect_image(path)
                    if verdict is not None:
                        exposures.append(f"{verdict.score:.2f}@{verdict.band}")
                        if not verdict.safe and not exempt:
                            held = self._quarantine(path)
                            # 带 blocked/ 前缀：被拦的图已经不在 cache/ 了，
                            # 只写文件名的话事后按日志去 cache 里根本找不到
                            blocked.append(
                                (f"blocked/{held.name}" if held else path.name, verdict)
                            )
                            logger.warning(
                                f"[comfyui] 拦下一张产出图：{verdict.reason}"
                                + (f"，已挪到 {held}" if held else "，但挪动失败")
                            )
                            continue

                    ok = await self._safe_send(
                        event, Image.fromFileSystem(str(path)), use_chain=True
                    )
                    sent += 1 if ok else 0
                    outputs.append(path.name)

                # ⚠️ 视频**没有**过内容兜底：`safety` 是逐帧像素判据，只认图片。
                # 目前这里只把文件名发给用户（视频文件本身没有投递），所以风险有限；
                # 但**如果以后要真的把视频发出去，必须同时补一个视频内容检查**，
                # 否则内容兜底就被绕过去了。见 README 第七章「已知缺口」。
                for item in result.videos:
                    try:
                        blob = await client.download(item)
                    except ComfyError as exc:
                        logger.warning(f"[comfyui] 下载视频失败：{exc}")
                        continue
                    path = self._save_cache(item, blob)
                    if path is None:
                        continue
                    await self._safe_send(
                        event, event.plain_result(f"视频已生成：{path.name}")
                    )
                    sent += 1

                elapsed = time.monotonic() - started
                summary_sizes = _summary_suffix(
                    summary, self._int("width", 0), self._int("height", 0)
                )
                extra = list(summary.get("preset_notes") or [])
                translated = summary.get("translated") or []
                if translated:
                    extra.append("中文已转标签：" + "、".join(cn for cn, _ in translated))
                leftover_cn = summary.get("leftover_cn") or ""
                if leftover_cn:
                    extra.append(f"未翻译中文：{leftover_cn}")

                # 日志里的参数列：尺寸/model/steps/denoise 各占一列，别混在一起
                samplers = summary.get("samplers") or []
                applied_denoise = denoise
                if applied_denoise is None and samplers:
                    raw_dn = samplers[0].get("denoise")
                    if isinstance(raw_dn, (int, float)) and mode != "controlnet":
                        applied_denoise = float(raw_dn)
                dims = summary.get("size")
                size_text = (
                    f"{dims[0]}x{dims[1]}" if isinstance(dims, (list, tuple)) and len(dims) == 2
                    else "-"
                )

                # ---- 落日志：聊天里不说的细节全在这里 ----
                self._log_draw(
                    guard_mod.DrawRecord(
                        status="ok" if sent else ("blocked" if blocked else "empty"),
                        user=str(event.get_sender_id() or ""),
                        chat=self._chat_label(event),
                        request=text,
                        preset=summary.get("preset") or "",
                        workflow=workflow.label,
                        reference=summary.get("reference") or "",
                        mode=mode,
                        denoise=f"{applied_denoise:.2f}" if applied_denoise is not None else "-",
                        size=size_text,
                        seed=str(summary.get("seed") if summary.get("seed") is not None else "-"),
                        elapsed=f"{elapsed:.1f}s",
                        output=";".join(outputs) or ";".join(n for n, _ in blocked),
                        exposure=";".join(exposures) or "-",
                        note=(
                            ("拦下 %d 张：" % len(blocked))
                            + ";".join(f"{n}({v.score:.2f})" for n, v in blocked)
                            if blocked
                            else " / ".join(extra) or "-"
                        ),
                    )
                )

                # ---- 回执：尽量短 ----
                if blocked:
                    # 台词写得含糊，不点破是什么内容 —— 群里说出来等于替对方广播
                    await self._speak(
                        event,
                        "blocked_image",
                        fallback="这张图被内容检查拦下了，没发出来。",
                    )
                    if sent == 0:
                        return
                if not sent:
                    if not blocked:
                        await self._safe_send(event, "图片下载或发送失败，见日志。")
                    return
                if self._bool("verbose_replies", False):
                    preset_note = (
                        f" + 预设「{summary['preset']}」" if summary.get("preset") else ""
                    )
                    extra_note = f"，{' / '.join(extra)}" if extra else ""
                    await self._safe_send(
                        event,
                        f"「{workflow.label}」出图完成{preset_note}，用时 {elapsed:.1f}s"
                        + summary_sizes
                        + extra_note,
                    )
                else:
                    # 短回执只带「用户会踩坑」的提醒：⚠ 开头的问题项 + 没翻出来的中文。
                    # 「参数按预设」「LoRA xxx」这类流水账留在日志里，不进聊天。
                    # 注意别再加前缀：这些问题项**本身就以 ⚠ 开头**，再加一个会
                    # 变成「⚠ ⚠ 工作流没有 LoRA 节点…」。
                    hints = [n for n in extra if n.startswith("⚠")]
                    if leftover_cn:
                        hints.append(f"⚠ 「{leftover_cn}」没翻成英文，下次直接写英文 tag")
                    if hints:
                        await self._safe_send(event, "；".join(hints))

    def _save_cache(self, item: dict[str, Any], blob: bytes) -> pathlib.Path | None:
        """把产出落到插件缓存目录，返回本地路径。"""
        filename = pathlib.Path(item.get("filename", "")).name
        if not filename:
            return None
        # 带节点/时间戳前缀，避免不同任务同名文件互相覆盖
        stamp = time.strftime("%Y%m%d-%H%M%S")
        target = self.cache_dir / f"{stamp}-{filename}"
        # 写之前确认目录在：cache_dir 只在 __init__ 建过一次，用户手动清 cache/
        # 省空间之后（这台机器 C 盘紧张时就会这么干），这里会一直写失败 ——
        # 而且表现是「出图了但发不出去」，很难往「目录没了」上想。
        # `_quarantine` 一直是自建目录的，两边保持一致。
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            logger.error(f"[comfyui] 建缓存目录失败：{exc}")
            return None
        try:
            target.write_bytes(blob)
        except OSError as exc:
            logger.error(f"[comfyui] 写缓存失败：{exc}")
            return None
        self._prune_cache()
        return target

    def _prune_cache(self) -> None:
        """缓存超过上限时删掉最旧的文件。"""
        limit = self._int("cache_limit", 200)
        if limit <= 0:
            return
        try:
            files = sorted(
                (p for p in self.cache_dir.iterdir() if p.is_file()),
                key=lambda p: p.stat().st_mtime,
            )
        except OSError:
            return
        for path in files[: max(0, len(files) - limit)]:
            try:
                path.unlink()
            except OSError:
                pass

    async def _safe_send(self, event: AstrMessageEvent, content: Any, *, use_chain: bool = False) -> bool:
        """发消息；任何网络异常只记日志，不让后台任务崩掉。"""
        try:
            if use_chain:
                from astrbot.api.event import MessageChain

                await event.send(MessageChain(chain=[content]))
            else:
                await event.send(event.plain_result(str(content)))
            return True
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"[comfyui] 发送消息失败：{exc}")
            return False

    async def _pick_text2img_workflow(
        self,
        requested: str,
        preset: presets_mod.Preset | None,
        text: str,
    ) -> Workflow:
        """丢弃参考图后，换回「能纯文生图」的工作流。

        两条约束互相拉扯，这个方法是用来同时满足它们的：

        1. **不能留 img2img 工作流**。它的画布尺寸来自取图节点（VAEEncode 的
           latent），没有参考图就退化成 768×768 并带上空控制图，出图直接崩坏
           （实测踩过两次）。
        2. **不能丢底模和 LoRA**。出装预设绑定的往往正是那份图生图工作流
           （底模 + 角色 LoRA 都在里面）。一旦换用一份不含 LoRA 的文生图工作流，
           角色直接变样 —— 实测同一套衣服的颜色都会跑掉（深粉渲染成紫色）。

        所以这里不是「排除预设的工作流」，而是**按模型签名找它的文生图孪生兄弟**：
        把加载器节点里的底模 + LoRA 抓出来当指纹，再找一个指纹相同、但不含取图
        节点的文生图工作流（实测同一套模型的图生图版与文生图版指纹完全一致）。
        """
        await self._object_info_cached()

        if requested:
            # 用户自己点名了工作流，尊重他的选择，不替它找孪生兄弟
            return self._resolve_workflow(requested)

        # 预设绑定的工作流：先看它本身能不能纯文生图，不行就按指纹找孪生文生图版
        preset_workflow: Workflow | None = None
        if preset is not None and preset.workflow:
            try:
                preset_workflow = self._resolve_workflow(preset.workflow)
            except ComfyError:
                preset_workflow = None

        text2img: list[tuple[Workflow, dict]] = []  # 可纯文生图的候选（已带指纹）
        for workflow in self.workflows.values():
            try:
                api = self._converted(workflow)
            except Exception:  # noqa: BLE001 - 转换失败的工作流直接跳过
                continue
            if not any(
                e.get("class_type") in ("SaveImage", "PreviewImage") for e in api.values()
            ):
                continue  # 没有输出节点，跑了也拿不到图
            text2img.append((workflow, _model_signature(api)))

        def find(signature: dict) -> Workflow | None:
            """在文生图候选里找指纹完全一致的。"""
            for workflow, sig in text2img:
                if sig == signature:
                    return workflow
            return None

        if preset_workflow is not None:
            try:
                preset_api = self._converted(preset_workflow)
            except Exception:  # noqa: BLE001
                preset_api = None
            if preset_api is not None and not uigraph.is_img2img(preset_api):
                return preset_workflow  # 预设工作流本来就是文生图，直接用
            if preset_api is not None:
                twin = find(_model_signature(preset_api))
                if twin is not None:
                    logger.info(
                        f"[comfyui] 去掉参考图：预设工作流「{preset_workflow.label}」"
                        f"是图生图，改用同模型的文生图工作流「{twin.label}」"
                    )
                    return twin
                logger.warning(
                    f"[comfyui] 去掉参考图：预设工作流「{preset_workflow.label}」"
                    "是图生图，且找不到同底模/LoRA 的文生图工作流，"
                    "只能退回不含 LoRA 的工作流（角色还原度会下降）"
                )
            else:
                # 连转换都失败，至少别把 LoRA 丢了，让上层去报错
                return preset_workflow

        # 预设没绑定或没找到孪生版本：退到全局默认，再退到任意文生图工作流
        configured = str(self._get("default_workflow", "") or "").strip()
        if configured:
            try:
                workflow = self._resolve_workflow(configured)
                api = self._converted(workflow)
                if not uigraph.is_img2img(api) and any(
                    e.get("class_type") in ("SaveImage", "PreviewImage")
                    for e in api.values()
                ):
                    return workflow
            except Exception:  # noqa: BLE001
                logger.warning(f"[comfyui] 默认工作流「{configured}」不可用于文生图")
        if text2img:
            return text2img[0][0]
        raise ComfyError("找不到可用的文生图工作流（需要含 EmptyLatentImage 与 SaveImage）")

    # ------------------------------------------------------------ 展示类命令

    async def _list_workflows(self, event: AstrMessageEvent):
        if not self.workflows:
            yield event.plain_result(
                f"还没有工作流。\n把 ComfyUI 里跑通的工作流用菜单「导出」存成 JSON，"
                f"放进：\n{self.workflow_dir}\n然后发「/{self._command()} 重载」。"
            )
            return
        lines = [f"可用工作流（{len(self.workflows)} 个）："]
        for wf in self.workflows.values():
            note = ""
            if wf.key in self._workflow_errors:
                note = f" !! {self._workflow_errors[wf.key]}"
            elif wf.description:
                note = f" —— {wf.description}"
            fmt = "API" if wf.is_api_format else "界面"
            lines.append(f"· {wf.label}（{fmt} 格式）{note}")
        lines.append(f"\n用法：/{self._command()} -w 工作流名 描述")
        yield event.plain_result("\n".join(lines))

    async def _workflow_info(self, event: AstrMessageEvent, name: str):
        if not self.workflows:
            yield event.plain_result("还没有载入任何工作流。")
            return
        try:
            workflow = self._pick_workflow(name, "") if name else self._pick_workflow("", "")
        except ComfyError as exc:
            yield event.plain_result(str(exc))
            return
        try:
            if workflow._api is None:
                object_info = await self._object_info_cached()
                workflow._api = uigraph.convert(workflow.raw, object_info)
        except (ComfyError, uigraph.ConversionError) as exc:
            yield event.plain_result(f"这份工作流暂时不可用：{exc}")
            return

        info = uigraph.describe(workflow._api)
        lines = [f"工作流：{workflow.label}"]
        if workflow.description:
            lines.append(f"说明：{workflow.description}")
        lines.append(f"格式：{'API 格式' if workflow.is_api_format else '界面格式（自动转换）'}")
        lines.append(f"节点数：{info['nodes']}")
        if info["checkpoints"]:
            lines.append(f"底模：{'、'.join(str(c) for c in info['checkpoints'])}")
        if info["size"]:
            lines.append(f"默认尺寸：{info['size'][0]}x{info['size'][1]}")
        for sampler in info["samplers"]:
            parts = [f"步数 {sampler['steps']}" if sampler.get("steps") else ""]
            if sampler.get("cfg"):
                parts.append(f"CFG {sampler['cfg']}")
            if sampler.get("sampler_name"):
                parts.append(str(sampler["sampler_name"]))
            if sampler.get("scheduler"):
                parts.append(str(sampler["scheduler"]))
            lines.append("采样：" + " / ".join(p for p in parts if p))
        if workflow.text_slots:
            lines.append("文本槽位：" + "、".join(workflow.text_slots))
        lines.append(f"文件：{workflow.path}")
        yield event.plain_result("\n".join(lines))

    async def _list_refs(self, event: AstrMessageEvent):
        if not self.refs:
            yield event.plain_result(
                f"还没有参考图。\n把立绘 PNG 放进：\n{self.refs_dir}\n"
                f"然后发「/{self._command()} 重载」。\n"
                "透明背景会自动合成到白底，超大图会自动等比缩小。"
            )
            return
        lines = [f"可用参考图（{len(self.refs)} 张）："]
        for ref in self.refs.values():
            try:
                size = ref.path.stat().st_size // 1024
            except OSError:
                size = 0
            note = f" —— {ref.note}" if ref.note else ""
            lines.append(f"· {ref.key}（{size} KB）{note}")
        lines.append(f"\n用法：/{self._command()} -r 参考图 你的描述")
        lines.append(f"只重画不改内容：/{self._command()} -r 参考图")
        lines.append(f"改得多一点：/{self._command()} -r 参考图 -d 0.7 描述")
        lines.append(f"目录：{self.refs_dir}")
        yield event.plain_result("\n".join(lines))

    async def _list_presets(self, event: AstrMessageEvent):
        if not self.presets:
            yield event.plain_result(
                f"还没有预设。\n预设文件：{self._presets_path()}\n"
                f"在里面写角色/风格的固定提示词，之后说关键词就能套用。"
            )
            return
        lines = [f"可用预设（{len(self.presets)} 个）："]
        for preset in self.presets.values():
            bits = []
            if preset.workflow:
                bits.append(f"工作流 {preset.workflow}")
            if preset.description:
                bits.append(preset.description)
            names = "、".join(preset.all_names()[1:])  # 去掉 key 本身
            note = f" —— {'；'.join(bits)}" if bits else ""
            lines.append(f"· {preset.key}（{preset.display_name or preset.name}）{note}")
            if names:
                lines.append(f"    触发词：{names}")
        lines.append(f"\n用法：/{self._command()} -p 预设名 追加内容")
        lines.append(f"不套预设：/{self._command()} -n 你的描述")
        lines.append(f"预设文件：{self._presets_path()}")
        yield event.plain_result("\n".join(lines))

    async def _preset_info(self, event: AstrMessageEvent, name: str):
        if not self.presets:
            yield event.plain_result("还没有载入任何预设。")
            return
        try:
            preset = self._resolve_preset(name) if name else self._pick_preset("", "")
        except ComfyError as exc:
            yield event.plain_result(str(exc))
            return
        if preset is None:
            yield event.plain_result("没指定预设名，用「/" + self._command() + " 预设」看列表")
            return

        lines = [f"预设：{preset.key}（{preset.display_name or preset.name}）"]
        if preset.description:
            lines.append(f"说明：{preset.description}")
        if preset.aliases:
            lines.append(f"触发词：{'、'.join(preset.aliases)}")
        if preset.workflow:
            lines.append(f"指定工作流：{preset.workflow}")
        if preset.lora:
            lines.append(f"LoRA：{preset.lora}")
        params = []
        if preset.width and preset.height:
            params.append(f"{preset.width}x{preset.height}")
        if preset.steps:
            params.append(f"{preset.steps} 步")
        if preset.cfg:
            params.append(f"CFG {preset.cfg}")
        if preset.sampler_name:
            params.append(preset.sampler_name)
        if preset.scheduler:
            params.append(preset.scheduler)
        if params:
            lines.append("参数：" + " / ".join(params))

        if preset.base_prompt_file:
            lines.append(f"提示词文件：{preset.base_prompt_file}")
        base = presets_mod.resolve_prompt(preset, "", self._preset_base_dir())
        if preset.load_error:
            lines.append(f"警告：{preset.load_error}")
        lines.append(f"基础提示词（{len(base)} 字符）：")
        lines.append(base or "（空）")
        if preset.negative_prompt:
            lines.append(f"负向提示词：{preset.negative_prompt}")
        if preset.notes:
            lines.append(f"备注：{preset.notes}")
        yield event.plain_result("\n".join(lines))

    # ------------------------------------------------------------ 生命周期

    async def initialize(self) -> None:
        self._load_workflows()
        self._load_presets()
        self._load_refs()
        if not self._bool("enable", True):
            logger.info("[comfyui] 插件已禁用（enable=false）")
            return
        try:
            report = await self._check_server()
            logger.info(f"[comfyui] 已连接 {self._server()}：{report}")
        except ComfyError as exc:
            logger.warning(
                f"[comfyui] 暂时连不上 {self._server()}：{exc}（出图时会再试）"
            )

    async def terminate(self) -> None:
        for task in list(self._tasks):
            task.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()
        if self._client is not None:
            await self._client.close()
            self._client = None


def _split_rule(line: str) -> tuple[str, list[str]]:
    """把一行路由规则拆成 ``(工作流名, 关键词列表)``。

    分隔符：``:`` / ``：`` / ``->`` / ``=``。工作流名里允许有空格，所以先按
    分隔符切开，而不是按逗号切。
    """
    for sep in ("->", ":", "：", "="):
        if sep in line:
            name, _, rest = line.partition(sep)
            keywords = [
                k.strip().strip('"').strip("'")
                for k in rest.replace("，", ",").split(",")
                if k.strip()
            ]
            return name.strip().strip('"').strip("'"), keywords
    return line.strip(), []


def _resolve_data_dir() -> pathlib.Path:
    """定位插件数据目录。

    优先用 AstrBot 官方的 ``StarTools.get_data_dir``（返回绝对路径，不受
    当前工作目录影响）；它在少数加载方式下拿不到插件元信息会抛异常，
    此时退回相对路径。
    """
    try:
        return StarTools.get_data_dir(PLUGIN_NAME)
    except Exception:  # noqa: BLE001 - 回退路径不该掩盖真正的问题
        fallback = pathlib.Path("data") / "plugin_data" / PLUGIN_NAME
        logger.warning(f"[comfyui] 无法解析官方数据目录，退回相对路径：{fallback}")
        return fallback


def _parse_draw_args(text: str) -> tuple[str, str, bool, str, str, float | None, str]:
    """解析出图参数，**开关可以出现在任意位置**。

    支持::

        -w / --workflow   工作流名
        -p / --preset     预设名
        -r / --ref        参考图
        -c / --control    参考图只用来锁结构/姿势（ControlNet），不当底图重绘
        -d / --denoise    重绘幅度 0~1（越小越接近参考图）
        -n / --no-preset  本次不套预设

    例：``-r my_ref 换个表情 -d 0.7`` 与 ``-d 0.7 -r my_ref 换个表情`` 等价。

    为什么不能只认「开头的开关」：那样 ``-r x 描述 -d 0.7`` 里的 ``-d`` 会被
    当成提示词的一部分吞掉，而没人会猜到开关必须写在提示词前面。

    返回 ``(工作流名, 预设名, 跳过预设, 参考图, 提示词, denoise, 参考图模式)``，
    模式是 ``"controlnet"`` 或 ``""``（空表示按图生图）。
    """
    workflow = ""
    preset = ""
    reference = ""
    skip_preset = False
    denoise: float | None = None
    ref_mode = ""
    prompt_parts: list[str] = []

    tokens = _tokenize(text or "")
    index = 0
    while index < len(tokens):
        token = tokens[index]
        lowered = token.lower()
        index += 1

        if lowered in ("--no-preset", "-n"):
            skip_preset = True
            continue

        # 无值开关：只切换「参考图怎么用」，不消费后面的 token
        if lowered in ("--control", "-c"):
            ref_mode = "controlnet"
            continue

        flag = lowered if lowered in _VALUE_FLAGS else None
        if flag is None:
            prompt_parts.append(token)
            continue

        # 开关的值就是下一个 token（引号内的空格已在 _tokenize 里保留）。
        # 但如果下一个 token 本身是开关，说明这个开关没写值 —— 此时不能把它
        # 吃掉当值，否则 `-d -r x 描述` 会把 `-r` 当 denoise 的值，
        # 结果参考图丢失、`x 描述` 反而变成提示词。
        next_token = tokens[index] if index < len(tokens) else ""
        next_lower = next_token.lower()
        if (
            next_lower in _VALUE_FLAGS
            or next_lower in ("--no-preset", "-n", "--control", "-c")
        ):
            value = ""
        else:
            value = next_token
            index += 1
        if flag in ("-w", "--workflow"):
            workflow = value
        elif flag in ("-p", "--preset"):
            preset = value
        elif flag in ("-r", "--ref", "--reference"):
            reference = value
        elif flag in ("-d", "--denoise"):
            try:
                denoise = max(0.0, min(1.0, float(value)))
            except (TypeError, ValueError):
                denoise = None  # 值不合法就当没写，后面用默认值

    prompt_text = " ".join(prompt_parts).strip()
    return workflow, preset, skip_preset, reference, prompt_text, denoise, ref_mode


#: 需要跟一个值的开关。
_VALUE_FLAGS: frozenset[str] = frozenset(    {
        "-w",
        "--workflow",
        "-p",
        "--preset",
        "-r",
        "--ref",
        "--reference",
        "-d",
        "--denoise",
    }
)


def _tokenize(text: str) -> list[str]:
    """按空格切词，但保留引号内的整体（工作流名可能含空格）。"""
    tokens: list[str] = []
    buffer: list[str] = []
    quote = ""
    for char in text:
        if quote:
            if char == quote:
                quote = ""
            else:
                buffer.append(char)
            continue
        if char in ('"', "'"):
            quote = char
            continue
        if char.isspace():
            if buffer:
                tokens.append("".join(buffer))
                buffer = []
            continue
        buffer.append(char)
    if buffer:
        tokens.append("".join(buffer))
    return tokens


def _take_first_token(text: str) -> tuple[str, str]:
    """取出第一个「词」和剩余部分。

    工作流名可能含空格，所以优先按引号取；没有引号就取到第一个空格，
    再由 ``_resolve_workflow`` 做模糊匹配兜底。
    """
    stripped = text.strip()
    if not stripped:
        return "", ""
    for quote in ('"', "'"):
        if stripped.startswith(quote):
            end = stripped.find(quote, 1)
            if end > 0:
                return stripped[1:end], stripped[end + 1 :].strip()
    parts = stripped.split(None, 1)
    return parts[0], parts[1].strip() if len(parts) > 1 else ""


def _split_names(text: str) -> list[str]:
    """把模型给的一串名字拆成列表（容错：支持中英文逗号、顿号、空格）。"""
    if not text:
        return []
    cleaned = text
    for sep in ("，", "、", ";", "；", "|"):
        cleaned = cleaned.replace(sep, ",")
    return [part.strip().strip('"').strip("'") for part in cleaned.split(",") if part.strip()]


def _parse_workflow_flag(text: str) -> tuple[str, str]:
    """兼容旧调用：只解析 ``-w``。"""
    workflow, _preset, _skip, _ref, rest, _denoise, _mode = _parse_draw_args(text)
    return workflow, rest


#: 会改变姿势的关键词（中英都收）。
#: 为什么要这张表：img2img 的参考图会把姿势焊死（实测 denoise 0.45/0.65 都改不动），
#: 所以一旦用户要求换姿势，就该自动**去掉参考图**，否则提示词写了也白写。
#:
#: 刻意不收「坐」「卧」这类单字 —— 会误伤「坐标」「卧室」等无关词。
_POSE_KEYWORDS: tuple[str, ...] = (
    # 英文 danbooru 标签
    "sitting", "seiza", "kneeling", "on knees", "lying", "on back", "on stomach",
    "on side", "squatting", "crouching", "leaning", "hugging knees", "crossed legs",
    "legs crossed", "arms behind back", "walking", "running", "jumping", "bending",
    "looking back", "reaching out", "on bed", "on chair", "on floor", "w-sitting",
    # 中文（用两字以上的明确说法）
    "坐着", "坐下", "坐在", "坐姿", "坐好",
    "站起来", "站起身",
    "躺着", "躺下", "躺在床上", "趴着", "趴在", "跪下", "跪着",
    "蹲下", "蹲着", "抱膝", "盘腿", "翘起", "翘腿", "交叉腿",
    "靠着", "弯腰", "回头", "走着", "奔跑", "跳跃",
    # 动作类：不是「姿势」两个字，但同样会改变身体构图，参考图留着照样锁成站姿。
    #
    # ⚠️ 这组是**生产环境踩出来**的：用户发「校服该角色在黄昏的部室窗户旁看书」，
    # 连发两次中文 + 一次英文（Reading by the window…），三张全是站姿、没有书、
    # 没有窗户 —— 因为「看书」不在这张表里，预设自带的参考图被保留、姿势被焊死。
    # 「看书」明显需要双手持书、低头、改变上半身构图，必须算换姿势。
    "看书", "读书", "阅读", "看报", "写字", "写作业", "弹吉他", "弹钢琴",
    "reading", "reading book", "studying", "writing", "holding book",
    "holding a book", "holding a tablet",
)


def _wants_new_pose(text: str) -> bool:
    """判断用户是不是在要求换姿势。"""
    if not text:
        return False
    lowered = text.lower()
    return any(kw in lowered for kw in _POSE_KEYWORDS)


#: 会要求换**场景 / 背景**的关键词。
#:
#: 为什么单独一张表：参考图是**纯色背景的立绘**，denoise 0.45 下它不只锁姿势，
#: 也把背景一起锁成白底。实测用户写「…by the window in my office at dusk」
#: （英文，标签完全有效）出图依然是白底 —— 因为参考图在压着。
#: 所以要描述场景就必须去掉参考图。
#:
#: ⚠️ 为什么**不能**并进 ``_POSE_KEYWORDS``：并进去的话 ``_strip_pose_tags``
#: 会连预设里的 ``standing`` 一起摘掉。可「站在窗边」这种请求姿势本来就是站姿，
#: 摘掉 ``standing`` 会让姿势变成随机 —— 换场景 ≠ 换姿势。两者用途不同：
#:
#: * 换姿势 → 去掉参考图 **且** 摘掉预设里的姿势 tag
#: * 换场景 → 只去掉参考图，姿势 tag 保留
#:
#: 刻意不收「部屋」（会切坏「部屋着」这类预设名）、英文也不收光秃秃的 ``room``
#: —— 预设 key 里含 room 的（如居家服那类）会被误判成每次都在换场景。
_SCENE_KEYWORDS: tuple[str, ...] = (
    "教室", "部室", "走廊", "天台", "阳台", "庭院", "花园",
    "海边", "沙滩", "街道", "公园", "樱花", "雪景", "下雪", "雨天", "下雨",
    "黄昏", "傍晚", "夕阳", "日落", "夜晚", "晚上", "星空", "满月",
    "窗户", "窗边", "窗外", "窗台", "背景", "室内", "室外",
    "classroom", "club room", "bedroom", "hallway", "rooftop", "balcony",
    "garden", "beach", "street", "park", "cherry blossoms", "snowy",
    "dusk", "sunset", "twilight", "starry sky", "full moon",
    "windowsill", "by the window", "scenery", "background",
    "indoors", "outdoors",
)


def _wants_new_scene(text: str) -> bool:
    """判断用户是不是在描述场景 / 背景。"""
    if not text:
        return False
    lowered = text.lower()
    return any(kw in lowered for kw in _SCENE_KEYWORDS)


def _needs_free_composition(text: str) -> bool:
    """参考图是否挡路 —— 换姿势**或**换场景都得把它去掉。

    两个概念的用途不同（见 ``_SCENE_KEYWORDS`` 的注释），所以判定要分开；
    这个函数只是把「要不要丢参考图」这一件事合起来。
    """
    return _wants_new_pose(text) or _wants_new_scene(text)


#: 预设基础提示词里「把姿势焊死」的 tag。
#:
#: 为什么必须清掉：所有站姿出装预设的 base_prompt 结尾都写着字面量 ``standing``
#: （照官方立绘写的）。用户说「坐着看书」时，最终提示词会变成
#: ``…, red shoes, standing, 坐着看书`` —— ``standing`` 是模型认识的英文 tag，
#: 而中文多半被 CLIP 忽略，于是画出来还是站着。**光去掉参考图不够，姿势 tag
#: 也得一起清掉**，否则「自动换姿势」和「参考图锁姿势」是一个下场。
#:
#: 刻意不收 ``holding gun`` / ``handgun`` 这类道具 tag：那是出装的一部分，
#: 换姿势时不该把枪弄丢。
_POSE_PIN_TAGS: frozenset[str] = frozenset(
    {
        "standing",
        "sitting",
        "seiza",
        "kneeling",
        "on knees",
        "lying",
        "on back",
        "on stomach",
        "on side",
        "squatting",
        "crouching",
        "legs crossed",
        "crossed legs",
        "arms behind back",
        "walking",
        "running",
        "jumping",
        "bending",
        "looking back",
        "reaching out",
        "hand on own chest",
        "hands on own chest",
        "hand on own hip",
        "hands on hips",
        "arms at sides",
        "arm at side",
        "outstretched arms",
        "w-sitting",
        "indian style",
    }
)


def _strip_pose_tags(text: str) -> tuple[str, list[str]]:
    """从提示词里摘掉姿势 tag，返回 ``(新提示词, 被摘掉的 tag)``。

    支持带权重/括号的写法（``(standing:1.2)``）—— 先剥掉括号与权重再比对，
    否则预设里只要写了权重就会漏掉。
    """
    kept: list[str] = []
    removed: list[str] = []
    for part in (text or "").split(","):
        raw = part.strip()
        if not raw:
            continue
        probe = raw.strip()
        # 剥掉外层括号（可能套好几层）与 :权重
        while probe.startswith("(") and probe.endswith(")"):
            probe = probe[1:-1].strip()
        if ":" in probe:
            head, _, tail = probe.rpartition(":")
            if tail.strip().replace(".", "", 1).isdigit():
                probe = head.strip()
        if probe.lower() in _POSE_PIN_TAGS:
            removed.append(raw)
            continue
        kept.append(raw)
    return ", ".join(kept), removed


#: 标签之间的合法分隔符。翻译拼接时用来判断要不要补逗号。
_TAG_SEPARATORS = (",", "，", "、", " ", "(", "（", ")", "）", ";", "；", "|")

#: 中文说法 → 英文 danbooru 标签。
#:
#: 为什么需要这张表：底层 SDXL/Illustrious 只认 danbooru 标签，中文进 CLIP 基本
#: 是噪声。实测「坐着看书」进去、出图姿势纹丝不动。LLM 工具那条路可以让模型
#: 自己翻译，但 ``/画图 某套出装 坐着看书`` 是纯文本直通，没人翻译 —— 用户看到
#: 的就是「说了没用」。
#:
#: 所以这里内置一份常用说法，命中就**替换成英文标签**（同时丢掉中文原词，
#: 免得它占权重干扰）。表只求覆盖高频需求，长词优先匹配。
_CN_TAG_PHRASES: tuple[tuple[str, str], ...] = (
    # 姿势（长词在前，避免「坐着」把「坐着看书」吃掉）
    ("坐着看书", "sitting, reading book"),
    ("坐着阅读", "sitting, reading book"),
    ("坐着看手机", "sitting, holding phone"),
    ("坐在椅子上", "sitting, on chair"),
    ("坐在窗边", "sitting, window"),
    ("坐在床上", "sitting, on bed"),
    ("坐在地上", "sitting, on floor"),
    ("盘腿", "indian style, sitting"),
    ("抱膝", "hugging own legs, sitting"),
    ("坐着", "sitting"),
    ("坐下", "sitting"),
    ("坐姿", "sitting"),
    ("躺在床上", "lying, on back, on bed"),
    ("趴在床上", "lying, on stomach, on bed"),
    ("侧躺", "lying, on side"),
    ("躺着", "lying, on back"),
    ("趴着", "lying, on stomach"),
    ("睡觉", "sleeping, closed eyes"),
    ("跪下", "kneeling"),
    ("跪着", "kneeling"),
    ("蹲下", "squatting"),
    ("蹲着", "squatting"),
    ("站着", "standing"),
    ("站起来", "standing"),
    ("站起身", "standing"),
    ("弯腰", "bending over"),
    ("回头", "looking back"),
    ("走着", "walking"),
    ("奔跑", "running"),
    ("跳跃", "jumping"),
    ("翘腿", "crossed legs"),
    ("翘起", "crossed legs"),
    ("靠着", "leaning"),
    # 动作 / 道具
    ("看书", "reading book"),
    ("看手机", "holding phone"),
    ("抱猫", "holding cat, cat"),
    ("抱着猫", "holding cat, cat"),
    ("挥手", "waving"),
    ("举手", "arm up"),
    ("比心", "heart hands"),
    ("双手合十", "own hands together"),
    ("手放胸前", "hand on own chest"),
    # 表情
    ("微笑", "smile"),
    ("害羞", "embarrassed, blush"),
    ("脸红", "blush"),
    ("生气", "angry, pout"),
    ("哭泣", "crying, tears"),
    ("流泪", "tears"),
    ("惊讶", "surprised"),
    ("闭眼", "closed eyes"),
    ("眨眼", "one eye closed"),
    # 颜色（只收两字词：单字「黑」「白」会误伤「黑猫」「白天」等无关词）
    ("白色", "white"),
    ("黑色", "black"),
    ("红色", "red"),
    ("蓝色", "blue"),
    ("粉色", "pink"),
    ("紫色", "purple"),
    ("绿色", "green"),
    ("黄色", "yellow"),
    ("灰色", "grey"),
    ("金色", "gold"),
    ("银色", "silver"),
    ("棕色", "brown"),
    ("橙色", "orange"),
    ("天蓝色", "light blue"),
    ("浅蓝色", "light blue"),
    ("深蓝色", "navy blue"),
    # 服装 / 配饰（长词在前）
    ("过膝袜", "thighhighs"),
    ("长筒袜", "thighhighs"),
    ("吊带袜", "garter belt, thighhighs"),
    ("高跟鞋", "high heels"),
    ("连衣裙", "dress"),
    ("半身裙", "skirt"),
    ("长裙", "long skirt"),
    ("短裙", "miniskirt"),
    ("裙子", "skirt"),
    ("裤子", "pants"),
    ("校服", "school uniform"),
    ("水手服", "serafuku"),
    ("泳装", "swimsuit"),
    ("泳衣", "swimsuit"),
    ("比基尼", "bikini"),
    ("和服", "kimono"),
    ("浴衣", "yukata"),
    ("睡衣", "pajamas"),
    ("内衣", "underwear"),
    ("外套", "coat"),
    ("夹克", "jacket"),
    ("毛衣", "sweater"),
    ("衬衫", "shirt"),
    ("围裙", "apron"),
    ("披风", "cape"),
    ("斗篷", "cape"),
    ("短袜", "socks"),
    ("靴子", "boots"),
    ("皮鞋", "leather shoes"),
    ("眼镜", "glasses"),
    ("帽子", "hat"),
    ("围巾", "scarf"),
    ("手套", "gloves"),
    ("领带", "necktie"),
    ("蝴蝶结", "ribbon"),
    ("发饰", "hair ornament"),
    ("发带", "hairband"),
    ("耳环", "earrings"),
    ("项链", "necklace"),
    ("头发", "hair"),
    ("发型", "hair style"),
    # 场景 / 背景。
    # 生产环境用户写「在黄昏的部室窗户旁看书」，只有「看书」被翻出来，
    # 「在黄昏部室窗户旁」整段残留成噪声 —— 而场景恰恰是他这条请求的重点。
    #
    # 注意别收「部屋」：那是「居家服」预设名的前半段，会把预设名切坏
    # （预设名虽然会先被 strip_trigger 摘掉，但多一个坑不如少一个）。
    ("黄昏", "dusk"), ("傍晚", "evening"), ("夕阳", "sunset"), ("日落", "sunset"),
    ("夜晚", "night"), ("晚上", "night"), ("星空", "starry sky"), ("满月", "full moon"),
    ("窗户", "window"), ("窗边", "window"), ("窗外", "window"), ("窗台", "windowsill"),
    ("教室", "classroom"), ("部室", "club room"), ("走廊", "hallway"),
    ("天台", "rooftop"), ("阳台", "balcony"), ("庭院", "garden"), ("花园", "garden"),
    ("海边", "beach"), ("沙滩", "beach"), ("街道", "street"), ("公园", "park"),
    ("樱花", "cherry blossoms"), ("下雪", "snow"), ("雪景", "snowy scenery"),
    ("雨天", "rain"), ("下雨", "rain"),
    ("背景", "background"), ("室内", "indoors"), ("室外", "outdoors"),
)

#: 纯语气/动词填料词，翻不出有意义的标签，直接删掉。
#:
#: 为什么单独一张表：``换成白色长裙`` 里的「换成」对模型是纯噪声（实测留着会让
#: 出图跑偏，头发颜色都可能变），但删掉它不影响语义。删完可能留下空 tag
#: （``white, , long skirt``），``_compose_prompt`` 会按逗号切分并丢掉空片段。
#:
#: 长词在前：「给我画」必须先于「画」被删掉，否则会剩个孤零零的「给我」。
#:
#: ⚠️ **不收「一只」「一个」「一张」这类量词**：它们对模型确实是噪声，但删掉会改变
#: 用户原话，出问题时用户按原话在提示词里找不到自己写的内容，反而困惑。
#: 动词/助词（「换成」「的」「地」）不同 —— 它们只影响语法，删掉语义不变。
_CN_FILLER_WORDS: tuple[str, ...] = (
    "换成是",
    "换成",
    "换个",
    "改成",
    "变成",
    "给我画",
    "帮我画",
    "画一张",
    "画个",
    "我想要",
    "麻烦",
    "一下",
    "穿着",
    "戴着",
    "戴",
    "穿",
    "的",
    "地",
)


def _translate_cn_phrases(text: str) -> tuple[str, list[tuple[str, str]]]:
    """把中文需求翻成英文 danbooru 标签，返回 ``(新文本, 命中列表)``。

    只做子串替换，不碰已经是英文的部分。命中一个就把它替换掉，长词优先
    （``_CN_TAG_PHRASES`` 已经排好序），这样「坐着看书」不会被拆成
    「坐着」+「看书」两段重复标签。

    ⚠️ 替换时必须在**两侧补逗号**：中文常把两个动作直接连写（``躺在床上睡觉``），
    天真的 ``str.replace`` 会拼出 ``on bedsleeping`` 这种既不是英文也不是标签的
    垃圾 token，模型完全不认。所以拼接处要判一下前后是不是分隔符。
    """
    if not text:
        return text, []
    result = text
    applied: list[tuple[str, str]] = []
    # 先删填料词（「换成」这类），再翻有意义的词
    for filler in _CN_FILLER_WORDS:
        if filler in result:
            result = result.replace(filler, "")
    for cn, tags in _CN_TAG_PHRASES:
        if cn not in result:
            continue
        out: list[str] = []
        cursor = 0
        while True:
            hit = result.find(cn, cursor)
            if hit < 0:
                out.append(result[cursor:])
                break
            before = result[cursor:hit]
            out.append(before)
            if before and not before.rstrip().endswith(_TAG_SEPARATORS):
                out.append(", ")
            out.append(tags)
            cursor = hit + len(cn)
            nxt = result[cursor : cursor + 1]
            if nxt and nxt not in _TAG_SEPARATORS:
                out.append(", ")
        result = "".join(out)
        applied.append((cn, tags))
    return result, applied


def _leftover_cjk(text: str) -> str:
    """挑出翻译后仍然剩下的中文片段，用于提醒用户「这些词模型看不懂」。

    翻译表只覆盖高频说法。表外的中文（生僻动作、复杂构图）送进 CLIP 基本是噪声，
    有时不只是无效、还会把画面带偏（实测留着「换成白色长裙」能把头发染成粉色）。
    与其让用户以为「说了没用」，不如明确告诉他哪几个词没翻出来。
    """
    leftovers: list[str] = []
    current: list[str] = []

    def flush() -> None:
        if current:
            chunk = "".join(current).strip()
            if chunk:
                leftovers.append(chunk)
            current.clear()

    for ch in text or "":
        # 只认 CJK 汉字；日文假名/标点不算，免得把英文标签里的连字符之类误报
        if "\u4e00" <= ch <= "\u9fff":
            current.append(ch)
        else:
            flush()
    flush()
    # 去重且保持出现顺序
    seen: set[str] = set()
    unique: list[str] = []
    for chunk in leftovers:
        # 只报两字以上的片段。翻译掉「黄昏/窗户」这类词之后常剩下孤零零一个「在」，
        # 为了提示用户「「在」没翻成英文」而专门发一条消息纯属噪音 ——
        # 单字中文对画面的影响可以忽略，真正会带偏画面的是成串的实词。
        if len(chunk) < 2:
            continue
        if chunk not in seen:
            seen.add(chunk)
            unique.append(chunk)
    return "、".join(unique)


#: 加载器节点里「哪个字段是模型名」的映射。
#: 指纹只认这些，别的一律忽略 —— 换采样器、换尺寸都不该影响「这是不是同一个模型」。
_LOADER_FIELDS: dict[str, tuple[str, ...]] = {
    "CheckpointLoaderSimple": ("ckpt_name",),
    "CheckpointLoader": ("ckpt_name",),
    "UNETLoader": ("unet_name",),
    "CLIPLoader": ("clip_name",),
    "VAELoader": ("vae_name",),
    "LoraLoader": ("lora_name",),
    "LoraLoaderModelOnly": ("lora_name",),
}


def _model_signature(api: dict[str, Any]) -> dict[str, Any]:
    """抽出「这份工作流用的是什么底模 / 什么 LoRA」的指纹。

    用途：丢掉参考图后要把图生图工作流换成文生图工作流，但**不能顺手把底模和
    LoRA 也换掉**。比对指纹就能找到同一套模型、只是没有取图节点的文生图版本
    （实测同一套模型的两份工作流，图生图版与文生图版的指纹完全一致）。

    LoRA 的权重也进指纹：同一个 LoRA 文件 0.6 和 1.0 是两种画风，不该算同一个。
    """
    checkpoints: list[str] = []
    loras: list[str] = []
    for node in api.values():
        cls = node.get("class_type")
        for field in _LOADER_FIELDS.get(str(cls), ()):
            value = node.get("inputs", {}).get(field)
            if not value:
                continue
            if field == "lora_name":
                strength = node["inputs"].get("strength_model")
                loras.append(f"{value}@{strength}")
            else:
                checkpoints.append(str(value))
    return {
        "checkpoints": tuple(sorted(checkpoints)),
        "loras": tuple(sorted(loras)),
    }


def _summary_suffix(summary: dict[str, Any], width: int, height: int) -> str:
    bits: list[str] = []
    if width > 0 and height > 0:
        bits.append(f"{width}x{height}")
    elif summary.get("size"):
        bits.append(f"{summary['size'][0]}x{summary['size'][1]}")
    if summary.get("checkpoints"):
        bits.append(str(summary["checkpoints"][0]))
    if summary.get("samplers"):
        sampler = summary["samplers"][0]
        if sampler.get("steps"):
            bits.append(f"{sampler['steps']} 步")
    return ("（" + " / ".join(bits) + "）") if bits else ""


def _usage(command: str) -> str:
    return (
        f"用法：\n"
        f"· /{command} 一只白猫在窗台上 — 直接出图\n"
        f"· /{command} -p 预设名 穿着和服 — 用预设（角色/风格），追加你的描述\n"
        f"· /{command} -r 参考图名 换个表情 — 图生图：拿内置参考图当底图\n"
        f"· /{command} -r 参考图名 -c 换成和服 — 锁结构：只抄姿势，内容重画\n"
        f"· /{command} -r 参考图名 -d 0.7 描述 — 调重绘幅度（默认 0.55，越小越像参考图）\n"
        f"· /{command} -n 你的描述 — 这一张不套预设\n"
        f"· /{command} -w 工作流名 描述 — 指定工作流\n"
        f"· /{command} 参考图 — 看有哪些参考图\n"
        f"· /{command} 预设 — 看有哪些预设（角色/风格）\n"
        f"· /{command} 预设信息 预设名 — 看预设的提示词与参数\n"
        f"· /{command} 工作流 — 看有哪些工作流\n"
        f"· /{command} 日志 — 看最近的出图记录（时间/预设/参数/耗时/暴露度）\n"
        f"· /{command} 人设 — 看回执用的台词（存在 persona.json，可直接改）\n"
        f"· /{command} 状态 — 检查 ComfyUI 连通性\n"
        f"· /{command} 重载 — 重新扫描工作流、预设与参考图"
    )

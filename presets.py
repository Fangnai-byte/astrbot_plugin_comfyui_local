"""预设提示词：把「已经调好的角色/风格提示词」变成可复用、可追加的底稿。

为什么需要它
------------
工作流和提示词是一对搭档：LoRA 工作流必须配它训练时的触发词和角色 tag，
少一个就出不来那个角色。所以插件绝不能拿一句用户输入去**替换**整段正向
提示词——那是把辛苦调好的角色信息丢掉。

这里的模型是：

    最终正向提示词 = 预设基础提示词                     ← 角色还原的保证
                   + 用户的追加内容                     ← 每次想改的东西
                   + （可选）画质词

预设可以内联写 `base_prompt`，也可以写 `base_prompt_file` 指向一个文件
（例如你 ComfyUI 目录里已有的 `my_char_prompt.txt`），插件运行时读取，
这样你维护一个文件就够了，不用把提示词抄两遍。
"""

from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass, field
from typing import Any

#: 预设配置文件里允许出现的顶层键。
_ALLOWED_KEYS = {
    "name",
    "display_name",
    "description",
    "aliases",
    "workflow",
    "base_prompt",
    "base_prompt_file",
    "negative_prompt",
    "quality_tags",
    "width",
    "height",
    "steps",
    "cfg",
    "sampler_name",
    "scheduler",
    "lora",
    "notes",
    # 预留：按工作流分别给提示词，键是工作流名
    "workflow_prompts",
}


class PresetError(Exception):
    """预设配置有问题。"""


@dataclass
class Preset:
    """一套调好的出图配置。"""

    key: str
    name: str = ""
    display_name: str = ""
    description: str = ""
    aliases: list[str] = field(default_factory=list)
    workflow: str = ""
    base_prompt: str = ""
    base_prompt_file: str = ""
    negative_prompt: str = ""
    quality_tags: str = ""
    width: int = 0
    height: int = 0
    steps: int = 0
    cfg: float = 0.0
    sampler_name: str = ""
    scheduler: str = ""
    lora: str = ""
    notes: str = ""
    #: 图生图默认用哪张参考图（对应 refs/ 里的文件名，不含扩展名）。
    ref: str = ""
    #: 图生图重绘幅度（0~1），留 0 用全局默认。
    denoise: float = 0.0
    #: 参考图用法：``img2img``（当底图重绘）或 ``controlnet``（只取结构/姿势）。
    mode: str = ""
    #: union 模型的控制模式（auto/openpose/depth/canny...）。
    control_type: str = ""
    #: ControlNet 权重（0~2）。
    control_strength: float = 0.0
    workflow_prompts: dict[str, str] = field(default_factory=dict)
    #: 解析 base_prompt_file 失败时的原因（用于在聊天里提示）
    load_error: str = ""
    #: 已解析到的文件提示词缓存
    _file_prompt: str | None = None

    @property
    def label(self) -> str:
        return self.display_name or self.name or self.key

    def all_names(self) -> list[str]:
        return [n for n in (self.key, self.name, self.display_name, *self.aliases) if n]


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def parse_preset(key: str, raw: dict[str, Any]) -> Preset:
    """把一条 JSON 配置变成 Preset。未知键会被忽略（便于加注释用键）。"""
    if not isinstance(raw, dict):
        raise PresetError(f"预设「{key}」不是 JSON 对象")

    aliases = raw.get("aliases") or []
    if isinstance(aliases, str):
        aliases = [a.strip() for a in aliases.replace("，", ",").split(",") if a.strip()]
    elif isinstance(aliases, (list, tuple)):
        aliases = [str(a).strip() for a in aliases if str(a).strip()]
    else:
        aliases = []

    workflow_prompts = raw.get("workflow_prompts") or {}
    if not isinstance(workflow_prompts, dict):
        workflow_prompts = {}

    return Preset(
        key=key,
        name=str(raw.get("name") or ""),
        display_name=str(raw.get("display_name") or ""),
        description=str(raw.get("description") or ""),
        aliases=aliases,
        workflow=str(raw.get("workflow") or ""),
        base_prompt=str(raw.get("base_prompt") or ""),
        base_prompt_file=str(raw.get("base_prompt_file") or ""),
        negative_prompt=str(raw.get("negative_prompt") or ""),
        quality_tags=str(raw.get("quality_tags") or ""),
        width=_as_int(raw.get("width")),
        height=_as_int(raw.get("height")),
        steps=_as_int(raw.get("steps")),
        cfg=_as_float(raw.get("cfg")),
        sampler_name=str(raw.get("sampler_name") or ""),
        scheduler=str(raw.get("scheduler") or ""),
        lora=str(raw.get("lora") or ""),
        notes=str(raw.get("notes") or ""),
        ref=str(raw.get("ref") or ""),
        denoise=_as_float(raw.get("denoise")),
        mode=str(raw.get("mode") or ""),
        control_type=str(raw.get("control_type") or ""),
        control_strength=_as_float(raw.get("control_strength")),
        workflow_prompts={str(k): str(v) for k, v in workflow_prompts.items()},
    )


def load_presets(path: pathlib.Path) -> tuple[dict[str, Preset], str]:
    """读取预设文件。返回 ``(预设表, 错误信息)``；空错误串表示没问题。"""
    if not path.exists():
        return {}, ""
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {}, f"预设文件读取失败：{exc}"
    if not isinstance(raw, dict):
        return {}, "预设文件顶层必须是 JSON 对象"

    presets: dict[str, Preset] = {}
    problems: list[str] = []
    for key, value in raw.items():
        if key.startswith("_"):
            continue  # 以 _ 开头的是注释键
        try:
            presets[key] = parse_preset(key, value)
        except PresetError as exc:
            problems.append(str(exc))
    return presets, "；".join(problems)


def _read_text(path_text: str, *, base_dir: pathlib.Path) -> tuple[str, str]:
    """读取提示词文件，返回 ``(内容, 错误)``。相对路径按预设文件所在目录解析。"""
    candidate = pathlib.Path(path_text)
    if not candidate.is_absolute():
        candidate = base_dir / candidate
    try:
        text = candidate.read_text(encoding="utf-8")
    except OSError as exc:
        return "", f"读不到 {candidate}（{exc}）"
    return _extract_prompt_block(text), ""


def _is_tag_line(stripped: str) -> bool:
    """判断一行是不是纯粹的提示词 tag 行。

    这类手写提示词文件里，说明文字必然是中文、或带全角括号/破折号/冒号；
    而真正的 tag 行只有 ASCII 字母数字、空格和 ``(),:._-`` 这些权重语法字符。
    用这个判据比按关键字跳过可靠得多——例如「备选」那段里也夹着英文 tag，
    靠关键字根本拦不住。
    """
    if not stripped:
        return False
    if set(stripped) <= set("=-—_ "):
        return False  # 分隔线
    has_ascii_alpha = False
    for ch in stripped:
        if ch.isascii() and (ch.isalnum()):
            has_ascii_alpha = True
            continue
        if ch in " ,():._-+/[]{}'\"|":
            continue
        return False  # 出现中文/全角括号等，说明是说明行而非 tag
    return has_ascii_alpha


def _extract_prompt_block(text: str) -> str:
    """从带说明文字的提示词文件里抽出真正的正向提示词。

    这类文件通常长这样::

        ■ 正向提示词（LoRA 训练最准的造型）：

        my_char, (my_char:1.15), masterpiece, ...
        1girl, solo, white hair, ...

        （瞳色说明：……）

        （备选：想要粉和服版，把校服那串 tag 换成
          pink kimono, ... ——但 LoRA ...）

        ■ 负向提示词：
        worst quality, ...

    规则：定位「正向提示词」小节 → 只收纯 tag 行 → 遇到下一个 ``■``
    或连续空行结束。这样说明行和「备选」段落都不会被误收。
    """
    lines = text.splitlines()
    collected: list[str] = []
    started = False
    for line in lines:
        stripped = line.strip()
        if not started:
            # ⚠️ 不能一看到「正向提示词」就认为进入正文：说明文字里也会提到它
            # （例如「触发词必须放正向提示词最前面」）。先记下见到了这个说法，
            # 等下一个真正的分节标题（■ 开头）再正式开始收 tag。
            if "正向提示词" in stripped or "正向提示詞" in stripped:
                started = True
            continue
        if stripped.startswith(("■", "【")):
            if collected:
                break  # 提示词已收完，遇到下一节（通常是负向）
            # 还没收到任何东西 —— 这一行才是「正向提示词」小节标题本身，
            # 正文从它后面开始。
            continue
        if not stripped:
            if collected:
                break  # 空行 = 提示词段落结束
            continue
        if _is_tag_line(stripped):
            collected.append(stripped.rstrip(","))

    if not collected:
        # 没有小标题结构：退回宽松模式，仍然只收 tag 行
        for line in lines:
            stripped = line.strip()
            if _is_tag_line(stripped):
                collected.append(stripped.rstrip(","))

    return ", ".join(part for part in collected if part)


def resolve_prompt(preset: Preset, workflow_key: str, base_dir: pathlib.Path) -> str:
    """算出该预设在某工作流下应该用的基础提示词。

    优先级：该工作流的专属提示词 > 文件里的提示词 > 内联提示词。
    """
    per_workflow = preset.workflow_prompts.get(workflow_key)
    if per_workflow:
        candidate = pathlib.Path(per_workflow)
        if not candidate.is_absolute():
            candidate = base_dir / candidate
        if candidate.is_file():
            text, error = _read_text(per_workflow, base_dir=base_dir)
            if text:
                return text
            preset.load_error = error
        elif "\n" in per_workflow or "," in per_workflow:
            return per_workflow.strip()

    if preset.base_prompt_file:
        text, error = _read_text(preset.base_prompt_file, base_dir=base_dir)
        if text:
            return text
        preset.load_error = error

    return preset.base_prompt.strip()


def _normalize(text: str) -> str:
    """归一化名字：去掉标点/空白，方便「差不多的写法」也能匹配。

    LLM 和用户写预设名时标点很随意（``某套出装·手放胸前`` / ``某套出装(手放胸前)``
    / ``某套出装 手放胸前``），严格匹配会莫名其妙失败，所以比较时先把标点抹掉。
    """
    return "".join(ch for ch in (text or "").lower() if ch.isalnum())


def find_preset(presets: dict[str, Preset], wanted: str) -> Preset | None:
    """按 key / 名字 / 别名找预设。

    匹配顺序：精确 → 去标点后的精确 → 唯一子串。这样既容忍标点差异，
    又不会因为短名是长名的子串而误配（子串仅在唯一时接受）。
    """
    target = (wanted or "").strip()
    if not target:
        return None
    lowered = target.lower()
    norm = _normalize(target)
    if not norm:
        return None

    # 1) 精确（原样 / 忽略大小写 / 去标点）
    for preset in presets.values():
        for name in preset.all_names():
            if name.lower() == lowered or _normalize(name) == norm:
                return preset

    # 2) 唯一子串（两个方向都试）
    partial = [
        preset
        for preset in presets.values()
        if any(
            lowered in name.lower() or norm in _normalize(name)
            for name in preset.all_names()
        )
    ]
    if len(partial) == 1:
        return partial[0]
    return None


def find_by_trigger(presets: dict[str, Preset], text: str) -> Preset | None:
    """按用户输入里的关键词/别名挑预设。"""
    if not text:
        return None
    lowered = text.lower()
    best: Preset | None = None
    best_len = 0
    for preset in presets.values():
        for name in preset.all_names():
            if name and name.lower() in lowered and len(name) > best_len:
                best, best_len = preset, len(name)
    return best


def strip_trigger(preset: Preset, text: str) -> tuple[str, str]:
    """把触发用的预设名从用户输入里摘掉，返回 ``(剩下的描述, 摘掉的名字)``。

    为什么要摘：``/画图 某套出装坐着看书`` 里的 ``某套出装`` 只是用来选预设的，
    它本身不是提示词。留着的话会变成正向提示词里的一个中文 tag
    （``…, red shoes, 某套出装, sitting``）—— 而底层模型只认 danbooru 英文标签，
    中文纯属噪声 token，白占权重还可能干扰构图。

    摘**最长**的那个匹配名（``find_by_trigger`` 也按最长匹配选预设），保证
    ``某套出装腕差分前`` 不会被短名 ``某套出装`` 切一半留个尾巴。
    """
    if not text:
        return text, ""
    lowered = text.lower()
    best = ""
    for name in preset.all_names():
        if name and name.lower() in lowered and len(name) > len(best):
            best = name
    if not best:
        return text, ""
    cut = lowered.index(best.lower())
    rest = (text[:cut] + text[cut + len(best) :]).strip(" ,，、")
    return rest, best


def render_preset_file(presets: dict[str, Preset]) -> str:
    """把预设表渲染成一份可读的配置文本，方便用户直接在聊天里看/改。"""
    payload: dict[str, Any] = {}
    for key, preset in presets.items():
        entry: dict[str, Any] = {}
        for field_name in (
            "name",
            "display_name",
            "description",
            "aliases",
            "workflow",
            "base_prompt",
            "base_prompt_file",
            "negative_prompt",
            "quality_tags",
            "width",
            "height",
            "steps",
            "cfg",
            "sampler_name",
            "scheduler",
            "lora",
            "notes",
            "ref",
            "denoise",
            "mode",
            "control_type",
            "control_strength",
            "workflow_prompts",
        ):
            value = getattr(preset, field_name)
            if value not in ("", 0, 0.0, [], {}):
                entry[field_name] = value
        payload[key] = entry
    return json.dumps(payload, ensure_ascii=False, indent=2)

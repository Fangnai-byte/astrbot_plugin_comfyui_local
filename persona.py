"""人设台词：让回执像「该角色在说话」，而不是机器人播报。

用户的原话是「开始时让该角色回复一句，然后判断是否可以生图，这样不仅有个性
还有对于 r18 内容的保证」。所以这个模块解决两件事：

1. **个性** —— 开始出图时回一句有人味的台词，而不是 ``🎨 开始画了～``。
2. **把内容判定摆到台面上** —— 台词按**场景**分别写（该不该放行、是全年龄
   还是只有私聊能看），于是「过没过内容检查」这件事在聊天里就看得见，
   不用等出图后才知道被拦。

⚠️ 两个必须守住的约束
--------------------
* **一句就好**。用户之前明确要求过回执要短，所以每句台词都控制在
  一句话以内；技术细节（工作流/预设/参数/耗时）仍然只进 ``draw.log``。
* **群里拒绝时不要点破内容**。在群聊里回一句「这是 R18 内容，不能画」
  等于**当着所有人面把请求内容说出来**。所以 ``denied_prompt`` 的台词
  一律写得含糊（「这个我不太想画」），具体命中了什么词只写日志。

台词存在 ``data/plugin_data/.../persona.json``，随时可改，
改完发 ``/画图 重载`` 生效（不用重启）。文件里缺哪个场景就用内置默认，
所以以后新增场景不会把老文件搞坏。
"""

from __future__ import annotations

import json
import pathlib
import random
from typing import Any

#: 场景名 -> 台词列表。
#:
#: 场景说明：
#:
#: * ``start``          正常开始出图
#: * ``pose``           自动去掉了参考图（要换姿势）
#: * ``scene``          自动去掉了参考图（要换场景/背景）
#: * ``private_r18``    私聊 + R18 白名单：可以收露骨产出，台词单独一套
#: * ``queued``         前面还有任务在跑
#: * ``denied_prompt``  提示词没过内容门（**群里要含糊**）
#: * ``denied_rate``    触发频率限制
#: * ``blocked_image``  产出图没过内容门
DEFAULT_PERSONA: dict[str, list[str]] = {
    "start": [
        "好的，请稍等一下哦～",
        "嗯，我这就画……稍等片刻。",
        "交给我吧，马上就好～",
        "好的呀，等我一下下。",
        "嗯，开始画了，请稍等。",
    ],
    "pose": [
        "嗯，这次换个姿势……请稍等。",
        "好的，姿势我会改一下哦～",
        "那我重新构一下图，稍等片刻。",
        "嗯，换个姿势画，等我一下。",
    ],
    "scene": [
        "好的，换个场景试试……请稍等。",
        "嗯，这次到别的地方去～稍等一下。",
        "那我换个背景重新画，稍等。",
    ],
    "private_r18": [
        "好的……这里只有我们两个人在，我照做就是了。",
        "嗯……没有别人在，那我就画了。",
        "好吧，只给你一个人看的话……稍等。",
    ],
    "queued": [
        "前面还有一张在画，稍等一下哦～",
        "稍等呀，我手上还有一张没画完。",
    ],
    "denied_prompt": [
        "这个……我不太想画呢，换个说法好吗？",
        "唔……这个我不太方便画，换一个描述吧？",
        "抱歉，这个内容我还是不画了。",
    ],
    "denied_rate": [
        "稍微等一下再画好吗？我有点忙不过来……",
        "让我歇一口气好不好？过会儿再画。",
    ],
    "blocked_image": [
        "唔……这张画出来有点不太好，就先不发了。",
        "抱歉，这张好像不太合适，我先收起来了。",
    ],
}


class PersonaError(Exception):
    """人设文件有问题。"""


def default_persona() -> dict[str, list[str]]:
    """内置默认台词的副本。"""
    return {k: list(v) for k, v in DEFAULT_PERSONA.items()}


def load_persona(path: pathlib.Path) -> tuple[dict[str, list[str]], str]:
    """读取人设文件，返回 ``(台词表, 错误信息)``。

    文件不存在不算错误（用默认台词）；解析失败会返回错误信息但仍然用默认，
    不能让一个写坏的 JSON 把人设搞成空白 —— 那样机器人会一句话都不说。
    """
    persona = default_persona()
    if not path.exists():
        return persona, ""

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return persona, f"人设文件读取失败：{exc}"

    if not isinstance(raw, dict):
        return persona, "人设文件顶层必须是 JSON 对象"

    problems: list[str] = []
    for situation, lines in raw.items():
        if str(situation).startswith("_"):
            continue  # _ 开头是注释键
        if isinstance(lines, str):
            lines = [lines]
        if not isinstance(lines, list):
            problems.append(f"「{situation}」应该是字符串数组")
            continue
        cleaned = [str(x).strip() for x in lines if str(x).strip()]
        if cleaned:
            # 只覆盖文件里写了的场景，没写的继续用默认
            persona[str(situation)] = cleaned
    return persona, "；".join(problems)


def render_persona(persona: dict[str, list[str]], path: pathlib.Path | None = None) -> str:
    """给人看的台词表（``/画图 人设`` 用）。"""
    lines = ["当前人设台词："]
    for situation, items in persona.items():
        lines.append(f"· {situation}（{len(items)} 句）")
        for item in items:
            lines.append(f"    {item}")
    if path is not None:
        lines.append(f"\n想改就编辑：\n{path}\n改完发「/画图 重载」生效。")
    return "\n".join(lines)


def pick(
    persona: dict[str, list[str]], situation: str, **fields: Any
) -> str:
    """从某个场景里随机取一句，并做安全的占位符替换。

    未知占位符**不报错**：用户手改台词时很容易把 ``{extra}`` 写错成 ``{extras}``，
    为此让整条消息发不出去不值得 —— 替换不了就原样留在那里。
    """
    items = persona.get(situation) or DEFAULT_PERSONA.get(situation) or []
    if not items:
        return ""
    line = random.choice(items)
    if not fields:
        return line
    try:
        return line.format(**fields)
    except (KeyError, IndexError, ValueError):
        return line

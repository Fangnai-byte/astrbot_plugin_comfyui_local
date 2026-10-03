"""把 ComfyUI 的「界面格式」工作流（nodes/links）转换成「API 格式」prompt。

为什么需要这个模块
------------------
ComfyUI 有两种工作流 JSON：

* **界面格式**：ComfyUI 菜单里「导出 / Export」出来的，含 ``nodes`` / ``links``。
  用户平时在 ComfyUI 里保存、编辑的就是这种，也是 `user/default/workflows` 下的格式。
* **API 格式**：菜单里「导出（API）/ Export (API)」出来的，是 ``{节点id: {class_type, inputs}}``。
  ``POST /prompt`` **只接受这一种**。

让用户每次手动导出 API 格式很容易出错（拿错了就是一顿报错），所以本插件把
界面格式就地转换掉：用户直接把 ComfyUI 里能跑通的工作流丢进来就行。

转换里踩过的三个坑（都已处理，勿删）
------------------------------------
1. ``object_info`` 会给**连线输入**也标 ``widget=True``（例如 KSampler 的 ``model`` /
   ``positive``），所以不能靠这个标记区分「控件」和「插槽」。真正可靠的判据是
   声明类型：类型是 ``INT/FLOAT/STRING/BOOLEAN`` 或一个候选列表的才是控件，
   其他（MODEL / CONDITIONING / LATENT / IMAGE …）是插槽。
2. ``widgets_values`` 是**按位置**存的，而且 ``seed`` 后面会多出一个前端专用的
   ``control_after_generate`` 组合框值（"randomize"）。这个组合框不在节点
   schema 里，schema 只用 ``control_after_generate: true`` 提示它存在。
   少跳过这一格，后面所有参数都会整体错位一位（症状：seed 变成字符串、
   ``VAEDecode.samples`` 拿到错误的输出槽）。
3. 存档里的 ``links[][2]``（源节点输出槽）**可能是脏的**：ComfyUI 校验时用的是
   节点真实的 ``RETURN_TYPES``，所以一个 ``origin_slot`` 错位的工作流在网页端
   照样能跑，直接拿去提交 API 却会报 "tuple index out of range"。
   因此源槽位要按源节点声明的输出名重新定位，不能照抄。
"""

from __future__ import annotations

import random
from typing import Any

#: 只存在于网页前端、绝不能提交给 /prompt 的控件名。
FRONTEND_ONLY_WIDGETS: frozenset[str] = frozenset(
    {
        "control_after_generate",
        "control_before_generate",
        "control",
        "image_upload",
        "upload",
        "clip_upload",
        "mask_upload",
        "choose file to upload",
        "videoupload",
        "audio_upload",
        "audioUI",
        "model_upload",
        "lora_upload",
        "template",
    }
)

#: 判定「这是控件而不是插槽」的标量类型。
_SCALAR_TYPES: frozenset[str] = frozenset({"INT", "FLOAT", "STRING", "BOOLEAN", "COMBO"})


def is_widget_declaration(declared: Any) -> bool:
    """判断一个输入声明是不是前端控件（而非连线插槽）。

    两种形态都算控件：

    * **候选列表** —— 旧式写法，直接给出可选项，如 ``[["euler", "dpmpp_2m"], {}]``
    * **标量/COMBO 类型名** —— 新式写法，如 ``["COMBO", {"options": [...]}]``
      （``SetUnionControlNetType.type`` 就是这种）

    其余类型名（MODEL / CONDITIONING / LATENT / IMAGE / CONTROL_NET …）是插槽。
    """
    if isinstance(declared, list):
        return True
    return declared in _SCALAR_TYPES

#: 文本编码节点的类名，用于替换提示词。
_TEXT_NODE_TYPES: frozenset[str] = frozenset(
    {"CLIPTextEncode", "CLIPTextEncodeSDXL", "CLIPTextEncodeSDXLRefiner", "TextEncodeQwenImageEdit"}
)

#: 采样器类名，用于随机种子与步数统计。
_SAMPLER_TYPES: frozenset[str] = frozenset(
    {"KSampler", "KSamplerAdvanced", "KSamplerSelect", "SamplerCustom", "KSampler (Efficient)"}
)

#: 会加载模型文件的节点类型（合并式与分离式都算）。
_LOADER_TYPES: frozenset[str] = frozenset(
    {
        "CheckpointLoaderSimple",
        "CheckpointLoader",
        "unCLIPCheckpointLoader",
        "UNETLoader",
        "DiffusionModelLoader",
        "VAELoader",
        "CLIPLoader",
        "DualCLIPLoader",
        "TripleCLIPLoader",
        "LoraLoader",
        "LoraLoaderModelOnly",
        "ControlNetLoader",
        "UpscaleModelLoader",
        "StyleModelLoader",
        "GLIGENLoader",
        "PhotoMakerLoader",
        "IPAdapterModelLoader",
    }
)


class ConversionError(Exception):
    """界面格式工作流无法转换成 API 格式时抛出。"""


def is_api_format(workflow: Any) -> bool:
    """判断一份 JSON 是否已经是 API 格式。"""
    if not isinstance(workflow, dict) or not workflow:
        return False
    if "nodes" in workflow:
        return False
    sample = next(iter(workflow.values()))
    return isinstance(sample, dict) and "class_type" in sample


def _spec_config(spec: Any) -> dict[str, Any]:
    """取出输入声明的配置字典；没有就返回空字典。"""
    if isinstance(spec, list) and len(spec) >= 2 and isinstance(spec[1], dict):
        return spec[1]
    return {}


def _iter_declared_inputs(node_info: dict[str, Any]):
    """按声明顺序遍历 ``(名字, 声明类型, 配置)``，跳过重名。"""
    seen: set[str] = set()
    for section in ("required", "optional"):
        for name, spec in (node_info.get("input", {}).get(section) or {}).items():
            if name in seen:
                continue
            seen.add(name)
            if isinstance(spec, list) and spec:
                yield name, spec[0], _spec_config(spec)
            else:
                yield name, None, {}


def _widget_slots(node_info: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    """列出前端会写进 ``widgets_values`` 的控件槽位，顺序与序列化一致。

    ``seed`` 这类控件后面会额外跟一个前端专用的 ``control_after_generate``
    组合框，这里把它当成一个独立伪槽位产出，好让位置对齐。
    """
    slots: list[tuple[str, dict[str, Any]]] = []
    for name, declared, config in _iter_declared_inputs(node_info):
        if not is_widget_declaration(declared) or name in FRONTEND_ONLY_WIDGETS:
            continue
        slots.append((name, config))
        if config.get("control_after_generate"):
            slots.append(("control_after_generate", {}))
    return slots


def _input_keys(node_info: dict[str, Any]) -> tuple[dict[str, str], set[str]]:
    """返回 ``(显示标题 -> 真实输入名, 全部合法输入名)``。"""
    by_title: dict[str, str] = {}
    all_keys: set[str] = set()
    for name, _declared, config in _iter_declared_inputs(node_info):
        all_keys.add(name)
        title = config.get("title")
        if title:
            by_title.setdefault(title, name)
    return by_title, all_keys


def convert(workflow: dict[str, Any], object_info: dict[str, Any]) -> dict[str, Any]:
    """把界面格式工作流转换成 API prompt。已经是 API 格式则原样返回副本。"""
    if is_api_format(workflow):
        return {k: dict(v) for k, v in workflow.items()}

    nodes = workflow.get("nodes")
    if not isinstance(nodes, list):
        raise ConversionError("这份 JSON 既不是界面格式（缺 nodes）也不是 API 格式（缺 class_type）")

    by_id = {n.get("id"): n for n in nodes if isinstance(n, dict)}

    # link_id -> (源节点 id, 存档里的输出槽, 连接类型)
    raw_links: dict[Any, tuple[Any, int, str | None]] = {}
    for link in workflow.get("links") or []:
        if isinstance(link, dict):
            raw_links[link.get("id")] = (
                link.get("origin_id"),
                link.get("origin_slot", 0),
                link.get("type"),
            )
        elif isinstance(link, list) and len(link) >= 3:
            raw_links[link[0]] = (
                link[1],
                link[2],
                link[5] if len(link) > 5 else None,
            )

    slot_cache: dict[str, dict[str, int]] = {}

    def true_slot(src_node_id: Any, declared_slot: int, kind: str | None) -> int:
        """按源节点声明的输出名重新定位输出槽（存档槽位可能是脏的）。"""
        src_node = by_id.get(src_node_id)
        if src_node is None:
            return declared_slot
        class_type = src_node.get("type")
        if class_type not in slot_cache:
            info = object_info.get(class_type) or {}
            names = info.get("output_name") or info.get("output") or []
            mapping: dict[str, int] = {}
            for index, name in enumerate(names):
                mapping.setdefault(str(name), index)
            slot_cache[class_type] = mapping
        mapping = slot_cache[class_type]
        if kind is not None and kind in mapping:
            return mapping[kind]
        outputs = src_node.get("outputs") or []
        if isinstance(declared_slot, int) and 0 <= declared_slot < len(outputs):
            socket_type = outputs[declared_slot].get("type")
            if socket_type in mapping:
                return mapping[socket_type]
        return declared_slot

    prompt: dict[str, Any] = {}
    for node in nodes:
        if not isinstance(node, dict):
            continue
        # mode 2 = muted, 4 = bypassed；这两种节点 ComfyUI 不执行。
        if node.get("mode") in (2, 4):
            continue
        class_type = node.get("type")
        if not class_type:
            continue  # Note / Reroute 这类虚拟节点没有可执行类型
        info = object_info.get(class_type)
        if info is None:
            raise ConversionError(
                f"节点 {node.get('id')} 用了未知的 class_type「{class_type}」，"
                "说明它依赖的自定义节点在 ComfyUI 里没装"
            )

        inputs: dict[str, Any] = {}
        by_title, all_keys = _input_keys(info)

        # 1) 连线输入优先（连线一定盖过控件值）
        for socket in node.get("inputs") or []:
            if not isinstance(socket, dict):
                continue
            link_id = socket.get("link")
            if link_id is None:
                continue
            if link_id not in raw_links:
                raise ConversionError(f"节点 {node.get('id')} 引用了不存在的连线 {link_id}")
            raw_src, declared_slot, kind = raw_links[link_id]
            socket_name = socket.get("name", "")
            key = socket_name if socket_name in all_keys else by_title.get(socket_name, socket_name)
            inputs[key] = [str(raw_src), true_slot(raw_src, declared_slot, kind)]

        # 2) 控件值按位置对齐
        widgets = node.get("widgets_values")
        if isinstance(widgets, dict):
            for key, value in widgets.items():
                if key in FRONTEND_ONLY_WIDGETS or key in inputs:
                    continue
                inputs[key] = value
        elif isinstance(widgets, list) and widgets:
            slots = _widget_slots(info)
            # 先铺声明默认值：前端不会把某些控件写进 widgets_values
            # （例如 SaveImage.filename_prefix），缺了它后端会判定必填缺失。
            for name, config in slots:
                if name != "control_after_generate" and config.get("default") is not None:
                    inputs.setdefault(name, config["default"])
            cursor = 0
            for name, _config in slots:
                if cursor >= len(widgets):
                    break
                value = widgets[cursor]
                cursor += 1
                if name == "control_after_generate":
                    continue  # 前端专用，占位而已
                if not isinstance(inputs.get(name), list):
                    inputs[name] = value

        prompt[str(node["id"])] = {
            "class_type": class_type,
            "inputs": {k: v for k, v in inputs.items() if k in all_keys},
        }

    if not prompt:
        raise ConversionError("转换后没有任何可执行节点，工作流是空的吗？")
    return prompt


def find_text_nodes(prompt: dict[str, Any]) -> tuple[list[str], list[str]]:
    """找出「正面/负面」文本节点 id：顺着采样器的连线找，比按顺序猜可靠。

    ⚠️ **不能只看采样器的直接上游**。ControlNet 工作流里采样器的 `positive`
    指向的是 ``ControlNetApplyAdvanced`` 而不是 ``CLIPTextEncode``：真正的文本
    在它更上游。早先的实现到此为止，于是返回 ``(['15'], ['15'])``（同一个
    ControlNet 节点）—— 结果 ``apply_prompt`` 把提示词写进了 ControlNet 节点
    一个**不存在的 ``text`` 输入**，CLIPTextEncode 保持原样：

    * 用户提示词完全没到模型（换衣服、换表情的命令在 ControlNet 模式下静默失效）
    * 正面还先被负面覆盖（两个桶都指向同一个节点）

    所以这里改成从采样器**向上递归**，穿过只做条件透传的节点，直到找到真正的
    文本编码节点。
    """
    positives: list[str] = []
    negatives: list[str] = []
    for entry in prompt.values():
        if entry.get("class_type") not in _SAMPLER_TYPES:
            continue
        for key, bucket in (("positive", positives), ("negative", negatives)):
            ref = entry.get("inputs", {}).get(key)
            if not (isinstance(ref, list) and ref):
                continue
            for node_id in _trace_text_sources(prompt, str(ref[0]), key):
                if node_id not in bucket:
                    bucket.append(node_id)
    if positives or negatives:
        return positives, negatives
    # 兜底：没有采样器就按 id 顺序取文本节点，第一个当正面。
    texts = [
        node_id
        for node_id, entry in sorted(prompt.items(), key=lambda kv: _as_int(kv[0]))
        if entry.get("class_type") in _TEXT_NODE_TYPES
    ]
    return texts[:1], texts[1:]


#: 条件（conditioning）透传节点的输入端口：``类名 -> (正面口, 负面口)``。
#: 这些节点本身不含提示词，只是把上游的 conditioning 加工一下（贴 ControlNet、
#: 换区域、拼时间步…），所以要继续往上找。
_CONDITIONING_PASSTHROUGH: dict[str, tuple[str, str | None]] = {
    "ControlNetApplyAdvanced": ("positive", "negative"),
    "ControlNetApply": ("positive", None),
    "ControlNetApplySD3": ("positive", "negative"),
    "ControlNetApplySD3Advanced": ("positive", "negative"),
    "T2IAdapterApply": ("positive", None),
    "FreeU_V2": ("positive", "negative"),
}

#: 通用兜底：类名以此开头、且带 `conditioning` 输入的节点也当作透传。
_CONDITIONING_PREFIX = "Conditioning"


def _trace_text_sources(
    prompt: dict[str, Any], node_id: str, kind: str, _seen: set[str] | None = None
) -> list[str]:
    """从某个 conditioning 端口向上回溯，返回真正的文本编码节点 id。

    ``kind`` 是 ``"positive"`` / ``"negative"``，用来决定走透传节点的哪个输入口。
    回溯带环保护，找不到就返回空列表（调用方会退回兜底逻辑）。
    """
    seen = _seen if _seen is not None else set()
    if node_id in seen:
        return []
    seen.add(node_id)

    entry = prompt.get(node_id)
    if not isinstance(entry, dict):
        return []
    class_type = str(entry.get("class_type") or "")
    if class_type in _TEXT_NODE_TYPES:
        return [node_id]

    inputs = entry.get("inputs") or {}
    # 透传节点：优先走与正/负对应的那个口
    ports: list[str] = []
    mapped = _CONDITIONING_PASSTHROUGH.get(class_type)
    if mapped is not None:
        port = mapped[0] if kind == "positive" else mapped[1]
        if port:
            ports.append(port)
    elif class_type.startswith(_CONDITIONING_PREFIX):
        if kind in inputs:
            ports.append(kind)
        ports.extend(k for k in ("conditioning", "conditioning_1", "conditioning_2") if k in inputs)

    for port in ports:
        ref = inputs.get(port)
        if isinstance(ref, list) and ref:
            found = _trace_text_sources(prompt, str(ref[0]), kind, seen)
            if found:
                return found
    return []


def _as_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def apply_prompt(
    prompt: dict[str, Any],
    text: str,
    *,
    negative: str | None = None,
) -> None:
    """把用户提示词写进正面文本节点；给了 negative 就同时覆盖负面节点。"""
    positives, negatives = find_text_nodes(prompt)
    if not positives:
        raise ConversionError("工作流里找不到文本编码节点，无法注入提示词")
    for node_id in positives:
        prompt[node_id].setdefault("inputs", {})["text"] = text
    if negative:
        for node_id in negatives:
            prompt[node_id].setdefault("inputs", {})["text"] = negative


def set_latent_size(prompt: dict[str, Any], width: int, height: int) -> bool:
    """覆盖宽高（如果有 EmptyLatentImage 之类的节点）。返回是否改到了。"""
    changed = False
    for entry in prompt.values():
        inputs = entry.get("inputs")
        if not isinstance(inputs, dict):
            continue
        if "width" in inputs and "height" in inputs:
            inputs["width"] = int(width)
            inputs["height"] = int(height)
            changed = True
    return changed


def set_image_scale_size(prompt: dict[str, Any], width: int, height: int) -> bool:
    """把 ImageScale 节点改成指定尺寸，让控制图与画布比例一致。

    不做这一步的话：参考图是 632x1536、画布是 832x1216，ControlNet 会把
    参考图拉伸去适配画布，构图整体偏移（实测头部会被裁掉）。
    """
    changed = False
    for entry in prompt.values():
        if entry.get("class_type") != "ImageScale":
            continue
        inputs = entry.get("inputs")
        if not isinstance(inputs, dict):
            continue
        if "width" in inputs and "height" in inputs:
            inputs["width"] = int(width)
            inputs["height"] = int(height)
            inputs["crop"] = "disabled"  # 等比内缩，不裁切
            changed = True
    return changed


def fit_canvas(
    ref_width: int,
    ref_height: int,
    *,
    megapixels: float = 1.0,
    multiple: int = 8,
    min_side: int = 512,
    max_side: int = 2048,
) -> tuple[int, int]:
    """按参考图比例算出一个合适的画布尺寸。

    目标像素量默认 1.0MP（SDXL 的舒适区），并按 ``multiple`` 对齐 —— 与参考图
    同比例能让 ControlNet 的控制图原样生效，不会因为拉伸而错位。
    """
    if ref_width <= 0 or ref_height <= 0:
        return 832, 1216
    target_px = megapixels * 1_000_000
    scale = (target_px / (ref_width * ref_height)) ** 0.5
    width = ref_width * scale
    height = ref_height * scale
    # 夹到合理范围，再对齐
    longest = max(width, height)
    if longest > max_side:
        shrink = max_side / longest
        width, height = width * shrink, height * shrink
    shortest = min(width, height)
    if shortest < min_side:
        grow = min_side / shortest
        width, height = width * grow, height * grow

    def snap(value: float) -> int:
        return max(min_side, int(value) // multiple * multiple)

    return snap(width), snap(height)


def randomize_seeds(prompt: dict[str, Any], seed: int | None = None) -> None:
    """给每个采样节点换一个随机种子。"""
    for entry in prompt.values():
        inputs = entry.get("inputs")
        if not isinstance(inputs, dict):
            continue
        value = seed if seed is not None else random.randint(0, 2**63 - 1)
        if "seed" in inputs:
            inputs["seed"] = value
        if "noise_seed" in inputs:
            inputs["noise_seed"] = value


def set_sampler_params(
    prompt: dict[str, Any],
    *,
    steps: int | None = None,
    cfg: float | None = None,
    sampler_name: str | None = None,
    scheduler: str | None = None,
    denoise: float | None = None,
) -> bool:
    """覆盖采样参数。返回是否有任何一项被改到。

    只改工作流里**已经存在**的键：预设给了 scheduler 但工作流用的是
    KSamplerAdvanced（没有 scheduler 键）时，不该凭空塞一个进去。
    """
    changed = False
    for entry in prompt.values():
        if entry.get("class_type") not in _SAMPLER_TYPES:
            continue
        inputs = entry.get("inputs")
        if not isinstance(inputs, dict):
            continue
        for key, value in (
            ("steps", steps),
            ("cfg", cfg),
            ("sampler_name", sampler_name),
            ("scheduler", scheduler),
            ("denoise", denoise),
        ):
            if value is None or key not in inputs:
                continue
            # 数值型键做个简单校验，避免字符串进去让 ComfyUI 报类型错
            if key in ("steps", "cfg", "denoise"):
                try:
                    value = int(value) if key == "steps" else float(value)
                except (TypeError, ValueError):
                    continue
            inputs[key] = value
            changed = True
    return changed


def set_lora(prompt: dict[str, Any], lora_name: str) -> bool:
    """把工作流里 LoRA 节点的 lora_name 换成指定 LoRA。返回是否改到了。"""
    if not lora_name:
        return False
    changed = False
    for entry in prompt.values():
        inputs = entry.get("inputs")
        if not isinstance(inputs, dict) or "lora_name" not in inputs:
            continue
        inputs["lora_name"] = lora_name
        changed = True
    return changed


#: 能接收「参考图文件名」的输入键，按优先级排列。
_IMAGE_INPUT_KEYS = ("image", "image_base64", "image_url")


def is_img2img(prompt: dict[str, Any]) -> bool:
    """判断一个 prompt 是不是图生图（含取图节点）。"""
    return any(
        entry.get("class_type") in ("LoadImage", "ETN_LoadImageBase64", "LoadImageOutput")
        for entry in prompt.values()
    )


def set_input_image(prompt: dict[str, Any], image_name: str) -> bool:
    """给取图节点写入参考图。

    ``LoadImage`` 用的是 ComfyUI input 目录里的**文件名**；
    ``ETN_LoadImageBase64`` 这类第三方节点要的是 base64，所以按节点类型分流。
    返回是否找到了可写入的节点。
    """
    if not image_name:
        return False
    changed = False
    for entry in prompt.values():
        class_type = entry.get("class_type")
        inputs = entry.get("inputs")
        if not isinstance(inputs, dict):
            continue
        if class_type == "LoadImage":
            if "image" in inputs:
                inputs["image"] = image_name
                inputs.setdefault("upload", "image")
                changed = True
        elif class_type == "ETN_LoadImageBase64":
            if "image" in inputs:
                inputs["image"] = image_name
                changed = True
        else:
            continue
    return changed


def set_denoise(prompt: dict[str, Any], denoise: float) -> bool:
    """覆盖采样器的 denoise —— 图生图靠它控制「改多少」。"""
    return set_sampler_params(prompt, denoise=denoise)


#: 判断一个 prompt 是否含 ControlNet 应用节点。
_CONTROL_APPLY_TYPES = ("ControlNetApply", "ControlNetApplyAdvanced")


def has_controlnet(prompt: dict[str, Any]) -> bool:
    """判断 prompt 里有没有接 ControlNet。"""
    return any(entry.get("class_type") in _CONTROL_APPLY_TYPES for entry in prompt.values())


def set_control_strength(prompt: dict[str, Any], strength: float) -> bool:
    """设置 ControlNet 权重（0~2）。返回是否找到了应用节点。"""
    changed = False
    for entry in prompt.values():
        if entry.get("class_type") not in _CONTROL_APPLY_TYPES:
            continue
        inputs = entry.get("inputs")
        if isinstance(inputs, dict) and "strength" in inputs:
            inputs["strength"] = float(strength)
            changed = True
    return changed


def set_control_type(prompt: dict[str, Any], control_type: str) -> bool:
    """设置 union 模型的模式（``auto`` / ``openpose`` / ``depth`` / ...）。

    只有 union 系模型（带 ``SetUnionControlNetType`` 节点）才吃这个；
    专用模型（canny 模型、openpose 模型）没有该节点，此时静默跳过 —— 因为
    专用模型的模式是模型本身决定的，设不设都一样。
    """
    if not control_type:
        return False
    changed = False
    for entry in prompt.values():
        if entry.get("class_type") != "SetUnionControlNetType":
            continue
        inputs = entry.get("inputs")
        if isinstance(inputs, dict) and "type" in inputs:
            inputs["type"] = control_type
            changed = True
    return changed


def describe_controlnet(prompt: dict[str, Any]) -> dict[str, Any]:
    """抽出 ControlNet 相关摘要，用于回执与展示。"""
    info: dict[str, Any] = {"models": [], "types": [], "strength": None, "preprocessors": []}
    for entry in prompt.values():
        class_type = entry.get("class_type")
        inputs = entry.get("inputs") or {}
        if class_type in ("ControlNetLoader", "DiffControlNetLoader"):
            if inputs.get("control_net_name"):
                info["models"].append(inputs["control_net_name"])
        elif class_type == "SetUnionControlNetType":
            info["types"].append(inputs.get("type"))
        elif class_type in _CONTROL_APPLY_TYPES:
            info["strength"] = inputs.get("strength")
        elif class_type in PREPROCESSOR_TYPES:
            info["preprocessors"].append(class_type)
    return info


#: 能把图片转成 ControlNet 控制图的预处理器节点。
#: 前四个是 ComfyUI 核心自带；其余来自 comfyui_controlnet_aux（需另装）。
PREPROCESSOR_TYPES: frozenset[str] = frozenset(
    {
        "Canny",
        "CannyEdgePreprocessor",
        "DepthAnythingPreprocessor",
        "DepthAnythingV2Preprocessor",
        "Zoe-DepthMapPreprocessor",
        "MiDaS-DepthMapPreprocessor",
        "OpenposePreprocessor",
        "DWPreprocessor",
        "LineArtPreprocessor",
        "AnimeLineArtPreprocessor",
        "M-LSDPreprocessor",
        "MLSDPreprocessor",
        "HEDPreprocessor",
        "PiDiNetPreprocessor",
    }
)

#: 每个 union 模式「本该配哪类预处理器」——用来检测模式与预处理不匹配。
#: 例如工作流用 Canny 出边沿图、却把模式设成 depth，union 模型会把边沿图
#: 当深度图解读，结果不可预期。这种情况必须提醒，不能默默跑。
_MODE_EXPECTED_PREPROCESSORS: dict[str, frozenset[str]] = {
    "openpose": frozenset({"OpenposePreprocessor", "DWPreprocessor"}),
    "depth": frozenset(
        {
            "DepthAnythingPreprocessor",
            "DepthAnythingV2Preprocessor",
            "Zoe-DepthMapPreprocessor",
            "MiDaS-DepthMapPreprocessor",
        }
    ),
    "canny/lineart/anime_lineart/mlsd": frozenset(
        {"Canny", "CannyEdgePreprocessor", "LineArtPreprocessor",
         "AnimeLineArtPreprocessor", "M-LSDPreprocessor", "MLSDPreprocessor"}
    ),
    "hed/pidi/scribble/ted": frozenset({"HEDPreprocessor", "PiDiNetPreprocessor"}),
}


def check_controlnet_mode(prompt: dict[str, Any], control_type: str) -> str:
    """检查「union 模式」与「实际预处理器」是否匹配。返回警告文字，匹配则空串。"""
    if not control_type or control_type == "auto":
        return ""
    expected = _MODE_EXPECTED_PREPROCESSORS.get(control_type)
    if not expected:
        return ""
    info = describe_controlnet(prompt)
    actual = set(info["preprocessors"])
    if not actual:
        return ""
    if actual & expected:
        return ""
    return (
        f"模式设为「{control_type}」，但工作流用的预处理器是 "
        f"{'、'.join(sorted(actual))}，两者不匹配 —— "
        "union 模型会把控制图按错误的语义解读，结果不可预期。"
        f"请把预处理换成 {' / '.join(sorted(expected))}，"
        "或把模式改成 auto 让模型自己判断。"
    )


def describe(prompt: dict[str, Any]) -> dict[str, Any]:
    """从转换结果里抽一份摘要，用于在聊天里展示工作流信息。"""
    info: dict[str, Any] = {
        "nodes": len(prompt),
        "samplers": [],
        "checkpoints": [],
        "size": None,
        #: 采样器实际用的种子。写进 draw.log 后可以用来复现某一张图，
        #: 所以即使聊天里不显示也要抽出来。
        "seed": None,
    }
    for node_id, entry in sorted(prompt.items(), key=lambda kv: _as_int(kv[0])):
        class_type = entry.get("class_type")
        inputs = entry.get("inputs", {})
        if class_type in _LOADER_TYPES:
            for value in inputs.values():
                if isinstance(value, str) and value.lower().endswith(
                    (".safetensors", ".ckpt", ".pt", ".pth", ".gguf", ".sft", ".bin")
                ):
                    info["checkpoints"].append(value)
        if class_type in _SAMPLER_TYPES:
            info["samplers"].append(
                {
                    "node": node_id,
                    "steps": inputs.get("steps"),
                    "cfg": inputs.get("cfg"),
                    "sampler_name": inputs.get("sampler_name"),
                    "scheduler": inputs.get("scheduler"),
                    # 图生图真正的重绘幅度在这里，用户没显式给 -d 时也要能查
                    "denoise": inputs.get("denoise"),
                }
            )
            if info["seed"] is None and inputs.get("seed") is not None:
                info["seed"] = inputs.get("seed")
        if "width" in inputs and "height" in inputs and info["size"] is None:
            info["size"] = [inputs.get("width"), inputs.get("height")]
    return info


def missing_models(prompt: dict[str, Any], object_info: dict[str, Any]) -> list[str]:
    """提交前的模型预检。

    工作流引用了本机没装的模型时，ComfyUI 只会回一个很难懂的
    ``value_not_in_list``。这里提前把「哪个节点的哪个模型没装」说清楚。

    只检查枚举型输入（候选项是字符串列表的那种），所以不用给每种加载器
    硬编码字段名，同时天然避开步数、CFG 这类自由数值。
    """
    problems: list[str] = []
    for node_id, entry in sorted(prompt.items(), key=lambda kv: _as_int(kv[0])):
        class_type = entry.get("class_type")
        info = object_info.get(class_type) or {}
        for section in ("required", "optional"):
            for name, spec in (info.get("input", {}).get(section) or {}).items():
                if not isinstance(spec, list) or not spec:
                    continue
                choices = spec[0]
                if not (isinstance(choices, list) and choices):
                    continue
                if not all(isinstance(c, str) for c in choices):
                    continue
                value = (entry.get("inputs") or {}).get(name)
                # 连线输入是 list，枚举值一定不是，跳过所有非字符串
                if not isinstance(value, str):
                    continue
                if value not in choices:
                    problems.append(
                        f"节点 {node_id}（{class_type}）的 {name} = 「{value}」在本机不存在"
                    )
    return problems

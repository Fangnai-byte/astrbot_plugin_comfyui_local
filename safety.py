"""产出图的内容兜底：判断一张图是不是「太暴露」，以及提示词里有没有露骨内容。

两道门，可靠性差很多，**别搞混**：

1. **提示词门（可靠）** —— ``find_nsfw_keywords()``。用户写「裸」「nude」「内衣」
   这类词时**在生成之前**就拒绝。这是精确匹配，没有漏报，是真正的防线。
2. **图像门（不可靠的兜底）** —— ``scan_image_bytes()``。生成后按像素估个
   「暴露度」，超阈值就不发。下面全是它的实测数据和局限。

判据怎么来的（有实测依据）
--------------------------
环境里只有 PIL + numpy（没有 torch / onnxruntime），所以只能用像素判据。

第一版用经典 RGB 皮肤规则 `r>g>b 且 r-g>15`，**在该角色身上直接反向**：她整套
某套出装是粉色开衫，RGB 把粉色当皮肤，某套出装立绘的胸部带算出 0.87，比裸体还高。
换 HSV 就分开了（皮肤 H≈10~30°，粉色 H≈320~350°），纯白背景的官方立绘上
4 张裸 / 20 张出装**零错误**。

但真实出图不是纯白背景，第二版加了「扣掉背景」：``皮肤占比 - 背景占比``，
背景用四边环带的颜色直方图估计（一张图常有沙发+墙两种背景色，单一中位色盖不住）。

⚠️ 实测结论：**这个判据把类别排错了，所以没有哪个阈值能用**
--------------------------------------------------------
以下分值全部由 `safety_check.py` 调**本模块**跑出来（不是另写脚本手搓判据 ——
之前标定脚本和落地代码参数漂过，文档写 0.56、代码算 0.45，等于用错数据调阈值）。

官方立绘（纯白背景，24 张：4 裸 / 20 出装）—— 这是唯一的硬指标，**零错误**：

    裸的最低分    0.609
    出装的最高分  0.223
    间隔          +0.386   ← 干净

但真实出图不是纯白背景，同一判据在真实图上的排序就乱了：

    0.679  正常：坐姿看书（**开领较深，胸口那一段**算出的皮肤多）
    0.609  应拦：裸体立绘
    0.452  应拦：黑色比基尼
    0.403  正常：ControlNet 白某套出装（收窄横带后已不再误拦）
    0.223  正常：出装立绘（上界）

**正常图的分数高于裸体图，而比基尼又低于裸体图** —— 任何单一阈值都做不到
「拦下比基尼同时不误拦那张坐姿图」。根因是「皮肤面积」本身不区分
「裸体」和「可爱出装露大腿」，在这个角色身上两者面积相当。

默认阈值 0.55 是在这个乱序里做的取舍：**抓整屏是肉的明显事故，接受误拦少数
正常图**。标定时在一批真实出图缓存（155 张，会随出图持续增长）上拦下 **1 张**
（就是上面那张开领深的坐姿图）。比基尼那一档会漏。
每张图的实测分值都写进日志（`exposure=0.56 band=0.40-0.46`），
用户觉得太松/太紧直接调 `nsfw_skin_threshold`。

它确实抓到过真东西：有一次「坐在窗边」的生成跑出了**仰视裙底的构图**
（膝盖抬起、大腿张开、镜头从下往上），分值 0.89 被拦下并挪进 `blocked/` ——
那正是这套兜底要挡的「不好/不积极向上」的图，不是裸体也能抓到。
（同一条提示词换个种子就是正常图，说明出图的随机性正是需要这道门的原因。）

想升级成可靠的门（可选，需要装东西）
------------------------------------
装 ``onnxruntime`` + 一个 NSFW ONNX 模型（如 Marqo/nsfw-image-detection-384，
约 90MB，走 ``hf-mirror.com``），然后改 ``exposure_score`` 走模型推理即可。
现在的代码结构已经把它隔离在一个函数里，替换不影响调用方。
"""

from __future__ import annotations

import pathlib
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from PIL import Image

#: 皮肤色相区间（度）。橙黄方向；粉色在 320~350°，不会落进来。
_SKIN_HUE_LO = 5.0
_SKIN_HUE_HI = 40.0
#: 饱和度下限：太低是灰/白（白底、白发），不是皮肤。
_SKIN_SAT_LO = 0.10
#: 饱和度上限：太高的橙红更可能是衣服/特效，不是皮肤。
_SKIN_SAT_HI = 0.72
#: 明度下限：太暗是阴影/黑袜。
_SKIN_VAL_LO = 0.35

#: 躯干带的滑动窗口（相对整图高度）。**刻意到腰为止，不含大腿**。
#:
#: 上界从 0.56 收到 0.48 是**实测逼出来的**：ControlNet 出的出装立绘被误拦
#: （0.577），命中的带落在 0.54~0.60 —— 那一段是裙摆和黑过膝袜之间的
#: **大腿**（俗称绝对领域），是这套衣服本来的样子，不是暴露。
#: 收到 0.48 后同一张图降到 0.403（不再误拦），而裸体立绘仍是 0.609、
#: 真正该拦的仰视裙底图仍是 0.829 —— 两头都没丢，所以是纯改进。
_TORSO_LO = 0.20
_TORSO_HI = 0.48
_WINDOW = 0.06
#: 横向只看中间这一段：避开两侧的手臂和背景。
_TORSO_XLO = 0.28
_TORSO_XHI = 0.72

#: 背景直方图的量化级数（每通道）与采样环带宽度。
_BG_BINS = 16
_BG_RING = 0.06

#: 提示词里的露骨/不当内容关键词（中英都收）。
#: 命中且会话没有 R18 权限时，**在生成之前**就拒绝，省得白跑一张图。
#: 这是可靠的那道门，所以收得宽一点：多拦一次成本很低，漏放一张代价很大。
NSFW_PROMPT_KEYWORDS: tuple[str, ...] = (
    # 裸露
    "nude", "naked", "nsfw", "topless", "bottomless", "undressing", "undressed",
    "no bra", "nipples", "areola", "pussy", "genitals", "penis", "exposed breasts",
    "completely nude",
    "裸", "全裸", "裸体", "露出", "露点", "真空", "走光", "不穿",
    # 内衣/性暗示服装
    "lingerie", "underwear", "panties", "bikini", "see-through",
    "transparent clothing", "wet clothes", "nipple slip",
    "内衣", "内裤", "胸罩", "透视", "湿身", "比基尼", "情趣",
    # 性行为/情动
    "sex", "hentai", "explicit", "masturbation", "orgasm", "aroused", "horny",
    "porn", "cum", "ahegao",
    "情动", "发情", "性交", "做爱", "高潮", "色情", "淫",
    # 暴力血腥
    "guro", "gore", "bloodbath", "dismemberment", "decapitation", "snuff",
    "血腥", "猎奇", "肢解", "斩首",
)


class SafetyError(Exception):
    """安全检查本身出错（图片打不开等）。"""


@dataclass
class ImageVerdict:
    """一张图的检查结果。"""

    #: 是否判定为「可以发出去」
    safe: bool
    #: 暴露度分值（0~1），越高越暴露
    score: float
    #: 实际使用的阈值
    threshold: float
    #: 分值最高的那条横带（用于日志定位）
    band: str = ""
    #: 人类可读的说明
    reason: str = ""
    #: 附加信息（尺寸等），给日志用
    detail: dict[str, Any] = field(default_factory=dict)


def _to_rgb(image: Image.Image) -> Image.Image:
    """透明底合成到白底：立绘 PNG 带 alpha，不合成会被当成黑色像素。"""
    if image.mode in ("RGBA", "LA") or (
        image.mode == "P" and "transparency" in image.info
    ):
        rgba = image.convert("RGBA")
        canvas = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        return Image.alpha_composite(canvas, rgba).convert("RGB")
    return image.convert("RGB")


def _hsv(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """RGB uint8 -> (H 0~360, S 0~1, V 0~1)。全向量化，没有逐像素 Python 循环。"""
    rgb = arr.astype(np.float32) / 255.0
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    mx = rgb.max(axis=2)
    mn = rgb.min(axis=2)
    diff = mx - mn

    hue = np.zeros_like(mx)
    color = diff > 1e-6

    idx = color & (mx == r)
    hue[idx] = (60.0 * ((g[idx] - b[idx]) / diff[idx])) % 360.0
    idx = color & (mx == g)
    hue[idx] = 60.0 * ((b[idx] - r[idx]) / diff[idx]) + 120.0
    idx = color & (mx == b)
    hue[idx] = 60.0 * ((r[idx] - g[idx]) / diff[idx]) + 240.0

    sat = np.where(mx > 1e-6, diff / np.maximum(mx, 1e-6), 0.0)
    return hue, sat, mx


def skin_mask(arr: np.ndarray) -> np.ndarray:
    """RGB uint8 -> 皮肤掩码（bool）。用 HSV 而不是 RGB：粉色衣服会骗过 RGB。"""
    hue, sat, val = _hsv(arr)
    return (
        (hue >= _SKIN_HUE_LO)
        & (hue <= _SKIN_HUE_HI)
        & (sat >= _SKIN_SAT_LO)
        & (sat <= _SKIN_SAT_HI)
        & (val >= _SKIN_VAL_LO)
    )


def background_mask(
    arr: np.ndarray, bins: int = _BG_BINS, ring: float = _BG_RING
) -> np.ndarray:
    """用四边环带的颜色直方图估计背景，返回背景掩码。

    为什么不用「与背景中位色的距离」：一张图常有多种背景色（沙发 + 墙、
    天空 + 地），单一中位色盖不住。统计环带颜色的粗直方图，颜色落在环带里
    出现过的桶里的像素就算背景。
    """
    height, width = arr.shape[:2]
    ry = max(1, int(height * ring))
    rx = max(1, int(width * ring))
    border = np.concatenate(
        [
            arr[:ry].reshape(-1, 3),
            arr[-ry:].reshape(-1, 3),
            arr[:, :rx].reshape(-1, 3),
            arr[:, -rx:].reshape(-1, 3),
        ]
    )

    quant = (border.astype(np.int32) * bins) // 256
    flat = quant[:, 0] * bins * bins + quant[:, 1] * bins + quant[:, 2]
    counts = np.bincount(flat, minlength=bins**3)
    # 环带里出现得够多的颜色才算背景；太少的是主体探出去的边缘
    keep = np.nonzero(counts >= max(4, int(len(border) * 0.004)))[0]

    qa = (arr.astype(np.int32) * bins) // 256
    ia = qa[:, :, 0] * bins * bins + qa[:, :, 1] * bins + qa[:, :, 2]
    if not len(keep):
        return np.zeros(ia.shape, dtype=bool)
    return np.isin(ia, keep)


def exposure_score(image: Image.Image, max_side: int = 320) -> tuple[float, str]:
    """算暴露度：``皮肤占比 - 背景占比`` 在躯干带上的最大值。

    取滑动窗口的**最大值**而不是整段平均：只露胸或只露腰都能抓到，
    不会被大片裙子摊薄。

    返回 ``(分值, 最暴露的那条带)``。
    """
    img = _to_rgb(image)
    img.thumbnail((max_side, max_side), Image.LANCZOS)
    arr = np.asarray(img)
    height, width = arr.shape[:2]
    if height < 8 or width < 8:
        return 0.0, ""

    skin = skin_mask(arr)
    bg = background_mask(arr)

    x0, x1 = int(width * _TORSO_XLO), int(width * _TORSO_XHI)
    if x1 <= x0:
        x0, x1 = 0, width
    rows = np.maximum(
        skin[:, x0:x1].mean(axis=1) - bg[:, x0:x1].mean(axis=1), 0.0
    )

    span = max(2, int(height * _WINDOW))
    best = 0.0
    best_band = ""
    lo = _TORSO_LO
    while lo < _TORSO_HI:
        a = int(height * lo)
        b = min(height, a + span)
        if b > a:
            value = float(rows[a:b].mean())
            if value > best:
                best = value
                best_band = f"{lo:.2f}-{lo + _WINDOW:.2f}"
        lo = round(lo + 0.02, 4)
    return best, best_band


def _verdict(score: float, band: str, threshold: float, size: tuple[int, int]) -> ImageVerdict:
    safe = score < threshold
    reason = (
        f"暴露度 {score:.2f} < 阈值 {threshold:.2f}"
        if safe
        else f"暴露度 {score:.2f} ≥ 阈值 {threshold:.2f}（最暴露的一段在高度 {band}）"
    )
    return ImageVerdict(
        safe=safe,
        score=score,
        threshold=threshold,
        band=band,
        reason=reason,
        detail={"size": size},
    )


def scan_image_bytes(
    blob: bytes, *, threshold: float = 0.55, max_side: int = 320
) -> ImageVerdict:
    """检查内存里的图片字节。图片解不开时抛 ``SafetyError``，由调用方决定怎么办。"""
    import io

    try:
        with Image.open(io.BytesIO(blob)) as img:
            img.load()
            score, band = exposure_score(img, max_side=max_side)
            size = img.size
    except Exception as exc:  # noqa: BLE001
        raise SafetyError(f"图片打不开，无法检查：{exc}") from exc
    return _verdict(score, band, threshold, size)


def scan_image_file(
    path: pathlib.Path | str, *, threshold: float = 0.55, max_side: int = 320
) -> ImageVerdict:
    """检查磁盘上的一张图。"""
    try:
        with Image.open(path) as img:
            img.load()
            score, band = exposure_score(img, max_side=max_side)
            size = img.size
    except Exception as exc:  # noqa: BLE001
        raise SafetyError(f"图片打不开，无法检查：{exc}") from exc
    return _verdict(score, band, threshold, size)


def find_nsfw_keywords(text: str) -> list[str]:
    """在提示词里找露骨/不当关键词，返回命中的词（保持原顺序、去重）。

    只做子串匹配。**故意做得宽**：宁可多拦一次让用户换个说法，也不要漏放一张
    不该发的图出去。这道门是可靠的，成本又极低，所以收紧没有意义。
    """
    if not text:
        return []
    lowered = text.lower()
    hits: list[str] = []
    for kw in NSFW_PROMPT_KEYWORDS:
        if kw.lower() in lowered and kw not in hits:
            hits.append(kw)
    return hits

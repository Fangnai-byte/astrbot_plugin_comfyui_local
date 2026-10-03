"""参考图（img2img）支持。

为什么要这个
------------
官方立绘是**透明背景**的 PNG，而 ComfyUI 的 VAE 只吃 RGB。直接丢进去，
透明区域会变成黑块或脏边，所以上传前要先合成到纯色底。

另外立绘尺寸很大（958×2324 这种），直接送进 SDXL 会爆显存且构图会被拉扁，
所以按最长边等比缩到可控尺寸，并对齐到 8 的倍数（SDXL 的 VAE 下采样要求）。
"""

from __future__ import annotations

import hashlib
import io
import pathlib
from dataclasses import dataclass

#: 支持的参考图扩展名。
_REF_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}

#: ComfyUI input 目录里参考图的统一前缀，便于识别与覆盖。
UPLOAD_PREFIX = "astrbot_ref_"


class RefError(Exception):
    """参考图处理失败。"""


@dataclass
class Reference:
    """一张可作为 img2img 输入的参考图。"""

    key: str
    path: pathlib.Path
    note: str = ""

    @property
    def label(self) -> str:
        return self.note or self.key


def find_reference(refs: dict[str, Reference], wanted: str) -> Reference | None:
    """按 key 精确 / 大小写不敏感 / 唯一子串匹配找参考图。"""
    target = (wanted or "").strip()
    if not target:
        return None
    if target in refs:
        return refs[target]
    lowered = target.lower()
    exact = [r for k, r in refs.items() if k.lower() == lowered]
    if exact:
        return exact[0]
    partial = [r for k, r in refs.items() if lowered in k.lower()]
    if len(partial) == 1:
        return partial[0]
    return None


def load_refs(refs_dir: pathlib.Path) -> dict[str, Reference]:
    """扫描参考图目录。"""
    refs: dict[str, Reference] = {}
    if not refs_dir.is_dir():
        return refs
    for path in sorted(refs_dir.iterdir()):
        if path.is_file() and path.suffix.lower() in _REF_SUFFIXES:
            refs[path.stem] = Reference(key=path.stem, path=path)
    return refs


def load_notes(refs_dir: pathlib.Path) -> dict[str, str]:
    """读取可选的 refs.json（给参考图写说明）。"""
    import json

    meta = refs_dir / "refs.json"
    if not meta.is_file():
        return {}
    try:
        data = json.loads(meta.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(data, dict):
        return {}
    notes = data.get("notes")
    return {str(k): str(v) for k, v in notes.items()} if isinstance(notes, dict) else {}


def prepare_image(
    src: pathlib.Path,
    *,
    max_side: int = 1536,
    background: tuple[int, int, int] = (255, 255, 255),
) -> bytes:
    """把参考图合成到纯色底并等比缩放，返回 PNG 字节。

    透明像素不先合成的话，VAE 会把 alpha 当垃圾数据，出图出现黑边或色块。
    """
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover
        raise RefError("需要 Pillow 才能处理参考图（pip install Pillow）") from exc

    try:
        im = Image.open(src)
        im.load()
    except OSError as exc:
        raise RefError(f"打不开参考图 {src}：{exc}") from exc

    if im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info):
        rgba = im.convert("RGBA")
        canvas = Image.new("RGB", rgba.size, background)
        canvas.paste(rgba, mask=rgba.split()[-1])
        im = canvas
    else:
        im = im.convert("RGB")

    if max_side > 0 and max(im.size) > max_side:
        ratio = max_side / max(im.size)
        width = max(64, int(im.width * ratio))
        height = max(64, int(im.height * ratio))
        # 对齐到 8 的倍数：SDXL 的 VAE 要下采样 8 倍
        width -= width % 8
        height -= height % 8
        im = im.resize((max(64, width), max(64, height)), Image.LANCZOS)

    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return buf.getvalue()


def prepared_size(src: pathlib.Path, max_side: int = 1536) -> tuple[int, int]:
    """算出 ``prepare_image`` 之后的实际像素尺寸（不重算图像，只算尺寸）。

    插件用它按参考图比例定画布：画布与参考图同比例时，ControlNet 的控制图
    原样生效；比例不一致会被拉伸，构图整体偏移（实测会把头裁掉）。
    """
    max_side_cache: dict[pathlib.Path, tuple[int, int, int]] = {}
    key = src
    if key not in max_side_cache:
        try:
            from PIL import Image

            with Image.open(src) as im:
                max_side_cache[key] = (im.width, im.height, max_side)
        except Exception:  # noqa: BLE001 - 打不开就退回默认
            return 832, 1216
    width, height, limit = max_side_cache[key]
    if limit > 0 and max(width, height) > limit:
        ratio = limit / max(width, height)
        new_w = max(64, int(width * ratio))
        new_h = max(64, int(height * ratio))
        new_w -= new_w % 8
        new_h -= new_h % 8
        return max(64, new_w), max(64, new_h)
    return width, height


def upload_name(src: pathlib.Path, blob: bytes) -> str:
    """生成稳定且带短哈希的上传文件名。

    带哈希是为了：同一张图重复上传时能命中 ComfyUI 的缓存，改了图又不会
    和旧文件混淆（否则 ComfyUI 会一直用 input 目录里的旧图）。
    """
    digest = hashlib.sha1(blob).hexdigest()[:8]
    stem = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in src.stem)[:40]
    return f"{UPLOAD_PREFIX}{stem}_{digest}.png"

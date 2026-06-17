from __future__ import annotations

from functools import lru_cache
from pathlib import Path


_ICON_DIR = Path(__file__).resolve().parents[1] / "assets" / "icons"


def icon_path(name: str) -> Path:
    return _ICON_DIR / f"{name}.svg"


@lru_cache(maxsize=128)
def _read_svg(name: str) -> str:
    path = icon_path(name)
    return path.read_text(encoding="utf-8")


def inline_svg(
    name: str,
    *,
    size_px: int = 18,
    color: str = "currentColor",
    title: str | None = None,
) -> str:
    svg = _read_svg(name).strip()
    if title and "<title>" not in svg:
        tag_end = svg.find(">")
        if tag_end != -1:
            svg = svg[: tag_end + 1] + f"<title>{title}</title>" + svg[tag_end + 1 :]
    size_px = int(size_px)
    return (
        f'<span class="igp-svg" style="font-size:{size_px}px;'
        f'color:{color};display:inline-flex;line-height:1;">{svg}</span>'
    )


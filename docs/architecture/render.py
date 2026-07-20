#!/usr/bin/env python3
"""Render the StyleMatch method schema and export its browser data."""

from __future__ import annotations

import argparse
import html
import json
import math
from pathlib import Path
from textwrap import wrap

import yaml


def esc(value: object) -> str:
    return html.escape(str(value), quote=True)


def text_lines(value: str, width: int, limit: int = 2) -> list[str]:
    return wrap(value, width=width, break_long_words=False)[:limit]


def svg_text(lines: list[str], x: float, y: float, size: int, color: str, weight: int = 400) -> str:
    spans = "".join(
        f'<tspan x="{x}" dy="{0 if index == 0 else size + 5}">{esc(line)}</tspan>'
        for index, line in enumerate(lines)
    )
    return f'<text x="{x}" y="{y}" font-size="{size}" font-weight="{weight}" fill="{color}">{spans}</text>'


def edge_curve(source: dict, target: dict) -> tuple[tuple[float, float], ...]:
    source_right = source["x"] + source["w"]
    target_right = target["x"] + target["w"]
    if target["x"] >= source_right:
        start = (source_right, source["y"] + source["h"] / 2)
        end = (target["x"], target["y"] + target["h"] / 2)
        control = max(36, (end[0] - start[0]) * .48)
        return start, (start[0] + control, start[1]), (end[0] - control, end[1]), end
    if source["x"] >= target_right:
        start = (source["x"], source["y"] + source["h"] / 2)
        end = (target_right, target["y"] + target["h"] / 2)
        control = max(36, (start[0] - end[0]) * .48)
        return start, (start[0] - control, start[1]), (end[0] + control, end[1]), end
    moving_down = target["y"] >= source["y"]
    start = (source["x"] + source["w"] / 2, source["y"] + (source["h"] if moving_down else 0))
    end = (target["x"] + target["w"] / 2, target["y"] + (0 if moving_down else target["h"]))
    control = max(42, abs(end[1] - start[1]) * .46)
    direction = 1 if moving_down else -1
    return start, (start[0], start[1] + direction * control), (end[0], end[1] - direction * control), end


def render(schema: dict) -> str:
    meta = schema["meta"]
    canvas = meta.get("canvas", {})
    width = meta.get("canvas_width", canvas.get("width", 1800))
    height = meta.get("canvas_height", canvas.get("height", 900))
    nodes = {node["id"]: node for node in schema["nodes"]}
    categories = schema["categories"]
    parts = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<defs><marker id="arrow" markerWidth="10" markerHeight="10" refX="8" refY="5" orient="auto"><path d="M1 1 L9 5 L1 9Z" fill="#7d715a"/></marker></defs>',
        '<rect width="100%" height="100%" fill="#fffdf8"/>',
    ]

    for group in schema.get("groups", []):
        boxes = [nodes[node_id]["layout"] for node_id in group["nodes"]]
        x = min(box["x"] for box in boxes) - 26
        y = min(box["y"] for box in boxes) - 30
        right = max(box["x"] + box["w"] for box in boxes) + 26
        bottom = max(box["y"] + box["h"] for box in boxes) + 24
        parts.append(
            f'<rect x="{x}" y="{y}" width="{right-x}" height="{bottom-y}" rx="24" fill="{group["fill"]}" fill-opacity=".28" stroke="{group["stroke"]}" stroke-dasharray="5 8"/>'
        )
        parts.append(svg_text([group["label"]], x + 16, y + 20, 11, group["stroke"], 700))

    visible_edges = schema["edges"]
    if meta.get("diagram_edges") == "primary_only":
        visible_edges = [edge for edge in visible_edges if edge.get("role") == "primary"]

    for edge in visible_edges:
        source, target = nodes[edge["from"]]["layout"], nodes[edge["to"]]["layout"]
        start, control_a, control_b, end = edge_curve(source, target)
        dash = ' stroke-dasharray="7 7"' if edge["kind"] == "experimental" else ""
        color = {"primary": "#1f3a5f", "context": "#66745d"}.get(edge.get("role"), "#a38b55")
        parts.append(
            f'<path d="M{start[0]},{start[1]} C{control_a[0]},{control_a[1]} {control_b[0]},{control_b[1]} {end[0]},{end[1]}" fill="none" stroke="{color}" stroke-width="2.2" marker-end="url(#arrow)"{dash}/>'
        )

    for node in schema["nodes"]:
        box = node["layout"]
        category = categories[node["category"]]
        x, y, width, height = box["x"], box["y"], box["w"], box["h"]
        dash = ' stroke-dasharray="7 7"' if node.get("style") == "dashed" else ""
        text_color = "#1b2a44"
        note_color = "#6f6758"
        parts.append(f'<g id="node-{esc(node["id"])}">')
        parts.append(
            f'<rect x="{x}" y="{y}" width="{width}" height="{height}" rx="17" fill="{category["fill"]}" stroke="{category["stroke"]}" stroke-width="2"{dash}/>'
        )
        parts.append(svg_text(text_lines(node["label"], 28, 1), x + 20, y + 30, 19, text_color, 700))
        parts.append(svg_text(text_lines(node["signature"], 42, 2), x + 20, y + 56, 12, category["stroke"], 700))
        parts.append(svg_text(text_lines(node["note"], 46, 2), x + 20, y + height - 25, 11, note_color))
        parts.append("</g>")

    parts.append(svg_text([meta["title"]], 28, 28, 16, "#1f3a5f", 700))
    parts.append(svg_text([f'schema.yaml · v{meta["version"]} · {meta["updated"]}'], width - 262, 28, 11, "#8d816b", 600))
    parts.append("</svg>")
    return "\n".join(parts)


def web_payload(schema: dict) -> dict:
    return {
        key: schema.get(key, {})
        for key in ("meta", "categories", "nodes", "edges", "callouts")
    }


def render_png(schema: dict, path: Path) -> None:
    """Draw a dependency-free deterministic raster using Pillow."""
    from PIL import Image, ImageDraw, ImageFont

    scale = 2
    canvas = schema["meta"].get("canvas", {})
    width = schema["meta"].get("canvas_width", canvas.get("width", 1800))
    height = schema["meta"].get("canvas_height", canvas.get("height", 900))
    image = Image.new("RGB", (width * scale, height * scale), "#fffdf8")
    draw = ImageDraw.Draw(image)
    nodes = {node["id"]: node for node in schema["nodes"]}
    categories = schema["categories"]
    font_root = Path("/System/Library/Fonts/Supplemental")
    serif = font_root / "Georgia.ttf"
    serif_bold = font_root / "Georgia Bold.ttf"
    sans = font_root / "Arial.ttf"
    sans_bold = font_root / "Arial Bold.ttf"

    def font(path: Path, size: int):
        return ImageFont.truetype(str(path), size * scale)

    def xy(box: tuple[float, ...]) -> tuple[int, ...]:
        return tuple(round(value * scale) for value in box)

    def dashed_border(left: float, top: float, right: float, bottom: float, color: str) -> None:
        dash, gap = 9 * scale, 6 * scale
        x1, y1, x2, y2 = xy((left, top, right, bottom))
        for start in range(x1, x2, dash + gap):
            draw.line((start, y1, min(start + dash, x2), y1), fill=color, width=2 * scale)
            draw.line((start, y2, min(start + dash, x2), y2), fill=color, width=2 * scale)
        for start in range(y1, y2, dash + gap):
            draw.line((x1, start, x1, min(start + dash, y2)), fill=color, width=2 * scale)
            draw.line((x2, start, x2, min(start + dash, y2)), fill=color, width=2 * scale)

    for group in schema.get("groups", []):
        boxes = [nodes[node_id]["layout"] for node_id in group["nodes"]]
        left = min(box["x"] for box in boxes) - 26
        top = min(box["y"] for box in boxes) - 30
        right = max(box["x"] + box["w"] for box in boxes) + 26
        bottom = max(box["y"] + box["h"] for box in boxes) + 24
        draw.rounded_rectangle(xy((left, top, right, bottom)), radius=24 * scale, outline=group["stroke"], width=2 * scale)
        draw.text(xy((left + 16, top + 10)), group["label"], fill=group["stroke"], font=font(sans_bold, 11))

    visible_edges = schema["edges"]
    if schema["meta"].get("diagram_edges") == "primary_only":
        visible_edges = [edge for edge in visible_edges if edge.get("role") == "primary"]

    for edge in visible_edges:
        source, target = nodes[edge["from"]]["layout"], nodes[edge["to"]]["layout"]
        start, control_a, control_b, end = edge_curve(source, target)
        points = []
        for step in range(41):
            t = step / 40
            inverse = 1 - t
            x = inverse**3 * start[0] + 3 * inverse**2 * t * control_a[0] + 3 * inverse * t**2 * control_b[0] + t**3 * end[0]
            y = inverse**3 * start[1] + 3 * inverse**2 * t * control_a[1] + 3 * inverse * t**2 * control_b[1] + t**3 * end[1]
            points.append((round(x * scale), round(y * scale)))
        color = {"primary": "#1f3a5f", "context": "#66745d"}.get(edge.get("role"), "#a38b55")
        draw.line(points, fill=color, width=2 * scale, joint="curve")
        angle = math.atan2(points[-1][1] - points[-3][1], points[-1][0] - points[-3][0])
        size = 9 * scale
        arrow = [
            points[-1],
            (points[-1][0] - size * math.cos(angle - .55), points[-1][1] - size * math.sin(angle - .55)),
            (points[-1][0] - size * math.cos(angle + .55), points[-1][1] - size * math.sin(angle + .55)),
        ]
        draw.polygon(arrow, fill=color)

    for node in schema["nodes"]:
        box = node["layout"]
        category = categories[node["category"]]
        left, top = box["x"], box["y"]
        right, bottom = left + box["w"], top + box["h"]
        draw.rounded_rectangle(xy((left, top, right, bottom)), radius=17 * scale, fill=category["fill"])
        if node.get("style") == "dashed":
            dashed_border(left, top, right, bottom, category["stroke"])
        else:
            draw.rounded_rectangle(
                xy((left, top, right, bottom)), radius=17 * scale,
                outline=category["stroke"], width=2 * scale,
            )
        text_color = "#1b2a44"
        accent = category["stroke"]
        muted = "#6f6758"
        draw.text(xy((left + 20, top + 17)), node["label"], fill=text_color, font=font(serif_bold, 19))
        signature = "\n".join(text_lines(node["signature"], 42, 2))
        draw.multiline_text(xy((left + 20, top + 49)), signature, fill=accent, font=font(sans_bold, 11), spacing=2 * scale)
        note = "\n".join(text_lines(node["note"], 46, 2))
        draw.multiline_text(xy((left + 20, bottom - 42)), note, fill=muted, font=font(serif, 10), spacing=2 * scale)

    draw.text(xy((28, 12)), schema["meta"]["title"], fill="#1f3a5f", font=font(serif_bold, 13))
    image.resize((width, height), Image.Resampling.LANCZOS).save(path, "PNG", optimize=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--schema", default="docs/architecture/schema.yaml")
    parser.add_argument("--out-dir", default="docs/architecture")
    parser.add_argument("--web-js")
    parser.add_argument("--web-svg")
    parser.add_argument("--png", action="store_true")
    args = parser.parse_args()

    schema = yaml.safe_load(Path(args.schema).read_text(encoding="utf-8"))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    svg = render(schema)
    (out_dir / "diagram.svg").write_text(svg, encoding="utf-8")
    (out_dir / "diagram.html").write_text(
        "<!doctype html><meta charset='utf-8'><title>StyleMatch Method Flow</title>"
        "<style>body{margin:0;background:#fffdf8}svg{display:block;max-width:100%;height:auto;margin:auto}</style>" + svg,
        encoding="utf-8",
    )
    if args.web_svg:
        Path(args.web_svg).write_text(svg, encoding="utf-8")
    if args.web_js:
        Path(args.web_js).write_text(
            "// Generated from docs/architecture/schema.yaml. Do not edit by hand.\n"
            "window.STYLEMATCH_METHOD_FLOW = "
            + json.dumps(web_payload(schema), ensure_ascii=False, separators=(",", ":"))
            + ";\n",
            encoding="utf-8",
        )
    if args.png:
        render_png(schema, out_dir / "diagram.render.png")
        render_png(schema, out_dir / "diagram.png")


if __name__ == "__main__":
    main()

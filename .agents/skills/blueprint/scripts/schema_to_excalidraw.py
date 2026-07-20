#!/usr/bin/env python3
"""Export a blueprint schema to a simple editable Excalidraw scene.

This adapter is intentionally conservative: it preserves stable node IDs,
semantic colors, relationship labels, and a predictable layered layout. It is
meant as a human-editable collaboration handoff, not a replacement for the
schema or deterministic SVG renderer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import re
from typing import Any

try:
    import yaml
except ImportError as exc:  # pragma: no cover - runtime guidance
    raise SystemExit(
        "Missing dependency: pyyaml. Run with: uv run --with pyyaml python schema_to_excalidraw.py"
    ) from exc


CARD_W = 300
CARD_H = 118
LAYER_Y = {"top": 160, "middle": 360, "bottom": 580}
MARGIN_X = 90
GAP_X = 74


def load_schema(path: pathlib.Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def text_width(text: str, minimum: int = CARD_W) -> int:
    cjk = sum(1 for ch in text if ord(ch) > 127)
    latin = max(len(text) - cjk, 0)
    return max(minimum, min(420, 145 + cjk * 14 + latin * 8))


def sanitize_id(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "-", value).strip("-") or "item"


def compute_layout(nodes: list[dict[str, Any]]) -> dict[str, tuple[int, int, int, int]]:
    layers: dict[str, list[dict[str, Any]]] = {}
    for node in nodes:
        layers.setdefault(str(node.get("layer") or "middle"), []).append(node)

    layout: dict[str, tuple[int, int, int, int]] = {}
    ordered_layers = ["top", "middle", "bottom"] + [x for x in layers if x not in {"top", "middle", "bottom"}]
    for layer in ordered_layers:
        layer_nodes = layers.get(layer, [])
        if not layer_nodes:
            continue
        y = LAYER_Y.get(layer, max(LAYER_Y.values()) + 200)
        total_w = sum(text_width(str(n.get("label", ""))) for n in layer_nodes) + GAP_X * (len(layer_nodes) - 1)
        x = MARGIN_X if total_w > 1500 else max(MARGIN_X, int((1700 - total_w) / 2))
        for node in layer_nodes:
            explicit = node.get("layout") or {}
            w = int(explicit.get("w") or text_width(str(node.get("label", ""))))
            h = int(explicit.get("h") or CARD_H)
            nx = int(explicit.get("x") or x)
            ny = int(explicit.get("y") or y)
            layout[str(node["id"])] = (nx, ny, w, h)
            x += w + GAP_X
    return layout


def base_element(element_id: str, element_type: str, x: float, y: float) -> dict[str, Any]:
    seed = int(hashlib.sha256(element_id.encode("utf-8")).hexdigest()[:8], 16)
    nonce = int(hashlib.sha256(f"{element_id}:nonce".encode("utf-8")).hexdigest()[:8], 16)
    return {
        "id": element_id,
        "type": element_type,
        "x": x,
        "y": y,
        "angle": 0,
        "strokeColor": "#1f2937",
        "backgroundColor": "transparent",
        "fillStyle": "solid",
        "strokeWidth": 2,
        "strokeStyle": "solid",
        "roughness": 0,
        "opacity": 100,
        "groupIds": [],
        "frameId": None,
        "roundness": None,
        "seed": seed,
        "version": 1,
        "versionNonce": nonce,
        "isDeleted": False,
        "boundElements": None,
        "updated": 1,
        "link": None,
        "locked": False,
    }


def rectangle(element_id: str, x: int, y: int, w: int, h: int, fill: str, stroke: str, dashed: bool) -> dict[str, Any]:
    item = base_element(element_id, "rectangle", x, y)
    item.update(
        {
            "width": w,
            "height": h,
            "strokeColor": stroke,
            "backgroundColor": fill,
            "strokeStyle": "dashed" if dashed else "solid",
            "roundness": {"type": 3},
        }
    )
    return item


def text(element_id: str, x: int, y: int, value: str, size: int, color: str, width: int, weight: str = "normal") -> dict[str, Any]:
    item = base_element(element_id, "text", x, y)
    item.update(
        {
            "width": width,
            "height": max(size + 10, 24),
            "strokeColor": color,
            "backgroundColor": "transparent",
            "text": value,
            "fontSize": size,
            "fontFamily": 5,
            "textAlign": "left",
            "verticalAlign": "top",
            "containerId": None,
            "originalText": value,
            "lineHeight": 1.25,
        }
    )
    if weight == "bold":
        item["fontSize"] = size
    return item


def arrow(element_id: str, start: tuple[int, int], end: tuple[int, int], label: str = "", dashed: bool = False) -> list[dict[str, Any]]:
    sx, sy = start
    ex, ey = end
    item = base_element(element_id, "arrow", sx, sy)
    item.update(
        {
            "width": ex - sx,
            "height": ey - sy,
            "points": [[0, 0], [ex - sx, ey - sy]],
            "endArrowhead": "arrow",
            "startArrowhead": None,
            "strokeColor": "#64748b",
            "strokeStyle": "dashed" if dashed else "solid",
            "roundness": {"type": 2},
        }
    )
    elements = [item]
    if label:
        lx = int((sx + ex) / 2) - 65
        ly = int((sy + ey) / 2) - 24
        elements.append(text(f"{element_id}-label", lx, ly, str(label)[:30], 14, "#475569", 180))
    return elements


def export_scene(schema: dict[str, Any]) -> dict[str, Any]:
    nodes = schema.get("nodes", [])
    edges = schema.get("edges", [])
    callouts = schema.get("callouts", [])
    categories = schema.get("categories", {})
    layout = compute_layout(nodes)
    elements: list[dict[str, Any]] = []

    title = str((schema.get("meta") or {}).get("title") or "Architecture Diagram")
    subtitle = str((schema.get("meta") or {}).get("subtitle") or "schema.yaml derived Excalidraw handoff")
    elements.append(text("title", 90, 42, title, 28, "#0f172a", 820, "bold"))
    elements.append(text("subtitle", 90, 82, subtitle, 16, "#64748b", 980))

    for node in nodes:
        node_id = str(node["id"])
        x, y, w, h = layout[node_id]
        category = categories.get(node.get("category"), {})
        fill = str(category.get("fill") or "#f8fafc")
        stroke = str(category.get("stroke") or "#64748b")
        dashed = node.get("style") == "dashed"
        elements.append(rectangle(f"node-{sanitize_id(node_id)}", x, y, w, h, fill, stroke, dashed))
        elements.append(text(f"text-{sanitize_id(node_id)}-label", x + 18, y + 17, str(node.get("label") or node_id), 19, "#111827", w - 36, "bold"))
        if node.get("signature"):
            elements.append(text(f"text-{sanitize_id(node_id)}-sig", x + 18, y + 48, str(node.get("signature")), 13, stroke, w - 36))
        if node.get("note"):
            elements.append(text(f"text-{sanitize_id(node_id)}-note", x + 18, y + 73, str(node.get("note")), 12, "#475569", w - 36))
        for inset_index, inset in enumerate((node.get("insets") or [])[:6]):
            col = inset_index % 2
            row = inset_index // 2
            inset_w = max(96, int((w - 54) / 2))
            inset_h = 24
            inset_x = x + 18 + col * (inset_w + 12)
            inset_y = y + h - 18 - ((len((node.get("insets") or [])[:6]) + 1) // 2 - row) * 32
            elements.append(rectangle(f"inset-{sanitize_id(node_id)}-{inset_index}", inset_x, inset_y, inset_w, inset_h, "#ffffff", stroke, False))
            elements.append(text(f"text-inset-{sanitize_id(node_id)}-{inset_index}", inset_x + 8, inset_y + 6, str(inset), 12, "#475569", inset_w - 16))

    for index, callout in enumerate(callouts):
        callout_id = sanitize_id(str(callout.get("id") or f"callout-{index}"))
        label = str(callout.get("label") or callout_id)
        explicit = callout.get("layout") or {}
        target_id = str(callout.get("target") or "")
        if not target_id and callout.get("nodes"):
            target_id = str(callout.get("nodes", [None])[0])
        target = layout.get(target_id)
        w = int(explicit.get("w") or text_width(label, 170))
        h = int(explicit.get("h") or 38)
        if explicit.get("x") is not None:
            x = int(explicit["x"])
        elif target:
            tx, ty, tw, th = target
            x = tx + tw + 24 if tx + tw + 24 + w < 1650 else max(MARGIN_X, tx - w - 24)
        else:
            x = MARGIN_X + (index % 3) * 430
        if explicit.get("y") is not None:
            y = int(explicit["y"])
        elif target:
            tx, ty, tw, th = target
            y = ty + th + 18 + (index % 2) * 46
        else:
            y = 720 + index * 46
        fill = str(callout.get("fill") or "#ffffff")
        stroke = str(callout.get("stroke") or "#cbd5e1")
        elements.append(rectangle(f"callout-{callout_id}", x, y, w, h, fill, stroke, callout.get("style") == "dashed"))
        elements.append(text(f"text-callout-{callout_id}", x + 14, y + 10, label, 13, "#475569", w - 28))

    for index, edge in enumerate(edges):
        source = layout.get(str(edge.get("from")))
        target = layout.get(str(edge.get("to")))
        if not source or not target:
            continue
        sx, sy, sw, sh = source
        tx, ty, tw, th = target
        start = (sx + sw // 2, sy + sh)
        end = (tx + tw // 2, ty)
        if ty < sy:
            start = (sx + sw // 2, sy)
            end = (tx + tw // 2, ty + th)
        dashed = edge.get("style") == "dashed" or edge.get("kind") == "calibration"
        elements.extend(arrow(f"edge-{index}-{sanitize_id(str(edge.get('from')))}-{sanitize_id(str(edge.get('to')))}", start, end, str(edge.get("label") or ""), dashed))

    return {
        "type": "excalidraw",
        "version": 2,
        "source": "blueprint/schema_to_excalidraw.py",
        "elements": elements,
        "appState": {"viewBackgroundColor": "#ffffff", "gridSize": 20},
        "files": {},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--schema", default="schema.yaml")
    parser.add_argument("--out", default="diagram.excalidraw")
    args = parser.parse_args()

    scene = export_scene(load_schema(pathlib.Path(args.schema)))
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(scene, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {out}")
    print(f"elements: {len(scene['elements'])}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Deterministic architecture renderer for blueprint schemas.

Reads docs/architecture/schema.yaml and writes diagram.svg, diagram.html, and
optionally a named PNG when cairosvg is available.

Intended as a starter renderer. Copy this file into a project as
docs/architecture/render.py, then customize layout/style locally.
"""

from __future__ import annotations

import argparse
import html
import math
import pathlib
import textwrap
from dataclasses import dataclass
from typing import Any

try:
    import yaml
except ImportError as exc:  # pragma: no cover - runtime guidance
    raise SystemExit(
        "Missing dependency: pyyaml. Run with: uv run --with pyyaml python render.py"
    ) from exc


CANVAS_W = 1800
MARGIN_X = 92
TOP_Y = 175
ROW_GAP = 64
GROUP_PAD_X = 34
GROUP_PAD_Y = 48
GROUP_GAP = 56
LAYER_GAP = 120
GROUP_BOUNDARY_LAYER_GAP = 2 * GROUP_PAD_Y + GROUP_GAP
CARD_W = 300
CARD_H = 116
CARD_R = 18
MAX_PER_ROW = 4
CALLOUT_H = 34
INSET_H = 24

PATH_ROLE_WEIGHTS = {
    "primary": 8.0,
    "context": 4.0,
    "secondary": 2.0,
    "feedback": 1.0,
}
LANE_POSITIONS = {"left": 0.0, "center": 0.5, "right": 1.0}


@dataclass
class Box:
    id: str
    x: float
    y: float
    w: float
    h: float

    @property
    def cx(self) -> float:
        return self.x + self.w / 2

    @property
    def cy(self) -> float:
        return self.y + self.h / 2

    @property
    def top(self) -> float:
        return self.y

    @property
    def right(self) -> float:
        return self.x + self.w

    @property
    def bottom(self) -> float:
        return self.y + self.h


@dataclass
class RoutedEdge:
    source: Box
    target: Box
    color: str
    dash: str
    label: str
    points: list[tuple[float, float]]


def esc(value: Any) -> str:
    return html.escape("" if value is None else str(value), quote=True)


def load_schema(path: pathlib.Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def wrap_text(text: str, width: int) -> list[str]:
    if not text:
        return []
    chunks: list[str] = []
    for part in str(text).splitlines():
        if len(part) <= width:
            chunks.append(part)
        else:
            chunks.extend(textwrap.wrap(part, width=width, break_long_words=False) or [part])
    return chunks


def weighted_text_width(text: str) -> int:
    return sum(13 if ord(ch) > 127 else 7 for ch in str(text))


def layer_key(node: dict[str, Any]) -> str:
    return str(node.get("layer") or "middle")


def node_default_height(node: dict[str, Any]) -> float:
    title_lines = wrap_text(str(node.get("label", "")), 20)[:2]
    signature_lines = wrap_text(str(node.get("signature", "")), 32)[:2]
    note_lines = wrap_text(str(node.get("note", "")), 34)[:2]
    insets = [str(x) for x in (node.get("insets") or [])][:6]

    cursor = 68 if node.get("shape") == "cylinder" else 32
    cursor += 31 + max(0, len(title_lines) - 1) * 24
    if signature_lines:
        cursor += 22 + max(0, len(signature_lines) - 1) * 19
    if note_lines:
        cursor += 30 + max(0, len(note_lines) - 1) * 18
    if insets:
        cursor += 14
        rows = math.ceil(len(insets) / 2)
        cursor += rows * INSET_H + max(0, rows - 1) * 8
    return max(CARD_H, cursor + 18)


def default_canvas_width(nodes: list[dict[str, Any]]) -> int:
    layer_counts: dict[str, int] = {}
    for node in nodes:
        layer = layer_key(node)
        layer_counts[layer] = layer_counts.get(layer, 0) + 1
    widest_row = min(max(layer_counts.values(), default=1), MAX_PER_ROW)
    return max(1440, min(1920, widest_row * CARD_W + (widest_row + 1) * 120))


def composition_hints(
    composition: dict[str, Any],
) -> tuple[dict[frozenset[str], float], dict[str, float]]:
    edge_weights: dict[frozenset[str], float] = {}
    for path in composition.get("paths") or []:
        role = str(path.get("role") or "secondary")
        weight = PATH_ROLE_WEIGHTS.get(role, PATH_ROLE_WEIGHTS["secondary"])
        node_ids = [str(node_id) for node_id in path.get("nodes") or []]
        for source_id, target_id in zip(node_ids, node_ids[1:]):
            key = frozenset((source_id, target_id))
            edge_weights[key] = max(edge_weights.get(key, 0.0), weight)

    lane_positions: dict[str, float] = {}
    for lane in composition.get("lanes") or []:
        position = LANE_POSITIONS.get(str(lane.get("position") or "center"), 0.5)
        for node_id in lane.get("nodes") or []:
            lane_positions[str(node_id)] = position
    return edge_weights, lane_positions


def compute_layout(
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    canvas_w: float,
    groups: list[dict[str, Any]],
    composition: dict[str, Any] | None = None,
) -> dict[str, Box]:
    layer_order = ["top", "middle", "bottom"]
    layers: dict[str, list[dict[str, Any]]] = {name: [] for name in layer_order}
    for node in nodes:
        layers.setdefault(layer_key(node), []).append(node)

    node_groups: dict[str, set[str]] = {}
    for group in groups:
        group_id = str(group.get("id") or "")
        for node_id in group.get("nodes", []):
            node_groups.setdefault(str(node_id), set()).add(group_id)

    boxes: dict[str, Box] = {}
    current_y = TOP_Y
    ordered_layers = [
        layer for layer in layer_order + [x for x in layers if x not in layer_order] if layers.get(layer)
    ]
    edge_weights, lane_positions = composition_hints(composition or {})
    for layer in ordered_layers:
        layer_nodes = layers[layer]
        original_positions = {str(node["id"]): index for index, node in enumerate(layer_nodes)}
        if any(str(node["id"]) in lane_positions for node in layer_nodes):
            layers[layer] = sorted(
                layer_nodes,
                key=lambda node: (
                    lane_positions.get(str(node["id"]), 0.5),
                    original_positions[str(node["id"])],
                ),
            )

    for layer_index in range(1, len(ordered_layers)):
        previous_nodes = layers[ordered_layers[layer_index - 1]]
        current_layer = ordered_layers[layer_index]
        current_nodes = layers[current_layer]
        if any((node.get("layout") or {}).get("x") is not None for node in current_nodes):
            continue

        previous_positions = {str(node["id"]): index for index, node in enumerate(previous_nodes)}
        original_positions = {str(node["id"]): index for index, node in enumerate(current_nodes)}
        linked_positions: dict[str, list[tuple[int, float]]] = {
            str(node["id"]): [] for node in current_nodes
        }
        for edge in edges:
            source_id = str(edge.get("from") or "")
            target_id = str(edge.get("to") or "")
            weight = edge_weights.get(frozenset((source_id, target_id)), 1.0)
            if source_id in previous_positions and target_id in linked_positions:
                linked_positions[target_id].append((previous_positions[source_id], weight))
            elif target_id in previous_positions and source_id in linked_positions:
                linked_positions[source_id].append((previous_positions[target_id], weight))

        if not any(linked_positions.values()):
            continue

        previous_span = max(len(previous_nodes) - 1, 1)
        current_span = max(len(current_nodes) - 1, 1)

        def crossing_reduction_key(node: dict[str, Any]) -> tuple[float, int]:
            node_id = str(node["id"])
            linked = linked_positions[node_id]
            linked_weight = sum(weight for _, weight in linked)
            score = (
                sum(position * weight for position, weight in linked) / linked_weight
                if linked_weight
                else original_positions[node_id] * previous_span / current_span
            )
            if node_id in lane_positions:
                lane_score = lane_positions[node_id] * previous_span
                lane_weight = max(8.0, linked_weight * 2.0)
                score = (score * linked_weight + lane_score * lane_weight) / (
                    linked_weight + lane_weight
                )
            return score, original_positions[node_id]

        layers[current_layer] = sorted(current_nodes, key=crossing_reduction_key)

    for layer_index, layer in enumerate(ordered_layers):
        layer_nodes = layers.get(layer, [])
        row_count = math.ceil(len(layer_nodes) / MAX_PER_ROW)
        for row in range(row_count):
            row_nodes = layer_nodes[row * MAX_PER_ROW : (row + 1) * MAX_PER_ROW]
            span = canvas_w - 2 * MARGIN_X
            gap = (span - len(row_nodes) * CARD_W) / max(len(row_nodes) + 1, 1)
            row_heights: list[float] = []
            for index, node in enumerate(row_nodes):
                explicit = node.get("layout") or {}
                x = explicit.get("x")
                y = explicit.get("y")
                w = explicit.get("w", CARD_W)
                h = explicit.get("h")
                if h is None:
                    h = node_default_height(node)
                if x is None:
                    x = MARGIN_X + gap + index * (CARD_W + gap)
                if y is None:
                    y = current_y
                boxes[str(node["id"])] = Box(str(node["id"]), float(x), float(y), float(w), float(h))
                row_heights.append(float(h))
            current_y += max(row_heights, default=CARD_H)
            if row < row_count - 1:
                current_y += ROW_GAP
        layer_gap = LAYER_GAP
        if layer_index + 1 < len(ordered_layers):
            next_nodes = layers[ordered_layers[layer_index + 1]]
            current_groups = set().union(*(node_groups.get(str(node["id"]), set()) for node in layer_nodes))
            next_groups = set().union(*(node_groups.get(str(node["id"]), set()) for node in next_nodes))
            if current_groups and next_groups and current_groups.isdisjoint(next_groups):
                layer_gap = GROUP_BOUNDARY_LAYER_GAP
        current_y += layer_gap
    return boxes


def svg_text(
    lines: list[str],
    x: float,
    y: float,
    size: int,
    weight: str = "400",
    color: str = "#111827",
    anchor: str | None = None,
) -> str:
    anchor_attr = f' text-anchor="{anchor}"' if anchor else ""
    out = [
        f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" font-weight="{weight}" '
        f'fill="{color}"{anchor_attr}>'
    ]
    for i, line in enumerate(lines):
        dy = 0 if i == 0 else size + 6
        out.append(f'<tspan x="{x:.1f}" dy="{dy}">{esc(line)}</tspan>')
    out.append("</text>")
    return "\n".join(out)


def node_svg(node: dict[str, Any], box: Box, categories: dict[str, Any]) -> str:
    category = categories.get(node.get("category"), {})
    fill = category.get("fill", "#f9fafb")
    stroke = category.get("stroke", "#64748b")
    dash = ' stroke-dasharray="7 6"' if node.get("style") == "dashed" else ""
    shape = node.get("shape", "rect")

    parts: list[str] = [f'<g id="node-{esc(node["id"])}">']
    if shape == "cylinder":
        parts.append(
            f'<path d="M {box.x:.1f} {box.y+16:.1f} '
            f'C {box.x:.1f} {box.y-5:.1f}, {box.x+box.w:.1f} {box.y-5:.1f}, {box.x+box.w:.1f} {box.y+16:.1f} '
            f'L {box.x+box.w:.1f} {box.y+box.h-16:.1f} '
            f'C {box.x+box.w:.1f} {box.y+box.h+5:.1f}, {box.x:.1f} {box.y+box.h+5:.1f}, {box.x:.1f} {box.y+box.h-16:.1f} Z" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="2"{dash}/>'
        )
        parts.append(
            f'<ellipse cx="{box.cx:.1f}" cy="{box.y+16:.1f}" rx="{box.w/2:.1f}" ry="20" '
            f'fill="none" stroke="{stroke}" stroke-width="2"{dash}/>'
        )
    elif shape == "page":
        parts.append(
            f'<path d="M {box.x:.1f} {box.y:.1f} L {box.x+box.w-28:.1f} {box.y:.1f} '
            f'L {box.x+box.w:.1f} {box.y+28:.1f} L {box.x+box.w:.1f} {box.y+box.h:.1f} '
            f'L {box.x:.1f} {box.y+box.h:.1f} Z" fill="{fill}" stroke="{stroke}" stroke-width="2"{dash}/>'
        )
        parts.append(
            f'<path d="M {box.x+box.w-28:.1f} {box.y:.1f} L {box.x+box.w-28:.1f} {box.y+28:.1f} '
            f'L {box.x+box.w:.1f} {box.y+28:.1f}" fill="none" stroke="{stroke}" stroke-width="2"/>'
        )
    else:
        parts.append(
            f'<rect x="{box.x:.1f}" y="{box.y:.1f}" width="{box.w:.1f}" height="{box.h:.1f}" '
            f'rx="{CARD_R}" fill="{fill}" stroke="{stroke}" stroke-width="2"{dash}/>'
        )

    title_lines = wrap_text(str(node.get("label", "")), 20)
    signature_lines = wrap_text(str(node.get("signature", "")), 32)
    note_lines = wrap_text(str(node.get("note", "")), 34)
    insets = [str(x) for x in (node.get("insets") or [])]
    text_x = box.x + 22
    text_y = box.y + (68 if shape == "cylinder" else 32)
    parts.append(svg_text(title_lines[:2], text_x, text_y, 19, "700", "#111827"))
    next_y = text_y + 31 + max(0, len(title_lines[:2]) - 1) * 24
    if signature_lines:
        parts.append(svg_text(signature_lines[:2], text_x, next_y, 13, "700", stroke))
        next_y += 22 + max(0, len(signature_lines[:2]) - 1) * 19
    if note_lines:
        parts.append(
            f'<line x1="{text_x:.1f}" y1="{next_y:.1f}" x2="{box.x+box.w-22:.1f}" y2="{next_y:.1f}" '
            f'stroke="#e5e7eb" stroke-width="1"/>'
        )
        parts.append(svg_text(note_lines[:2], text_x, next_y + 22, 12, "400", "#4b5563"))
        next_y += 30 + max(0, len(note_lines[:2]) - 1) * 18
    if insets:
        rows = math.ceil(min(len(insets), 6) / 2)
        inset_height = rows * INSET_H + max(0, rows - 1) * 8
        inset_top = max(next_y + 14, box.y + box.h - 18 - inset_height)
        inset_w = (box.w - 54) / 2
        for index, inset in enumerate(insets[:6]):
            col = index % 2
            row = index // 2
            ix = box.x + 22 + col * (inset_w + 10)
            iy = inset_top + row * (INSET_H + 8)
            parts.append(
                f'<rect x="{ix:.1f}" y="{iy:.1f}" width="{inset_w:.1f}" height="{INSET_H}" '
                f'rx="8" fill="#ffffff" fill-opacity="0.78" stroke="{stroke}" stroke-opacity="0.55" stroke-width="1"/>'
            )
            parts.append(svg_text(wrap_text(inset, 18)[:1], ix + 9, iy + 16, 10, "700", "#475569"))
    parts.append("</g>")
    return "\n".join(parts)


def route_edge(
    edge: dict[str, Any],
    boxes: dict[str, Box],
    categories: dict[str, Any],
    nodes_by_id: dict[str, dict[str, Any]],
    route_offset: float = 0,
) -> RoutedEdge | None:
    source = boxes.get(str(edge.get("from")))
    target = boxes.get(str(edge.get("to")))
    if not source or not target:
        return None

    source_cat = categories.get(nodes_by_id[source.id].get("category"), {})
    color = source_cat.get("stroke", "#64748b")
    if edge.get("kind") == "orchestration":
        color = "#6b7280"
    dash = ' stroke-dasharray="8 7"' if edge.get("kind") == "calibration" or edge.get("style") == "dashed" else ""
    label = edge.get("label") or ""
    label_lines = wrap_text(str(label), 28)[:2] if label else []
    label_width = max(96, min(max((weighted_text_width(line) for line in label_lines), default=0) + 24, 220))
    label_height = 24 if len(label_lines) <= 1 else 42

    vertical_overlap = min(source.bottom, target.bottom) - max(source.top, target.top)
    same_layer = layer_key(nodes_by_id[source.id]) == layer_key(nodes_by_id[target.id]) and vertical_overlap > 12
    points_list: list[tuple[float, float]]
    if same_layer:
        moving_right = target.cx >= source.cx
        start_x = source.right if moving_right else source.x
        end_x = target.x if moving_right else target.right
        left = min(source.right, target.right)
        right = max(source.x, target.x)
        row_top = min(source.top, target.top)
        row_bottom = max(source.bottom, target.bottom)
        intervening = [
            box
            for box in boxes.values()
            if box.id not in {source.id, target.id}
            and box.x < right
            and box.right > left
            and box.y < row_bottom
            and box.bottom > row_top
        ]
        needs_label_lane = bool(label) and label_width > max(0, abs(end_x - start_x) - 16)
        if intervening or needs_label_lane:
            direction = 1 if moving_right else -1
            elbow = max(10, min(36, 18 - route_offset * 0.50))
            lane_distance = max(24, min(62, 42 + route_offset * 0.30))
            port_y_offset = (8 if moving_right else -8) + route_offset * 0.20
            lane_y = (
                max([source.bottom, target.bottom] + [box.bottom for box in intervening]) + lane_distance
                if moving_right
                else min([source.top, target.top] + [box.top for box in intervening]) - lane_distance
            )
            points_list = [
                (start_x, source.cy + port_y_offset),
                (start_x + direction * elbow, source.cy + port_y_offset),
                (start_x + direction * elbow, lane_y),
                (end_x - direction * elbow, lane_y),
                (end_x - direction * elbow, target.cy + port_y_offset),
                (end_x, target.cy + port_y_offset),
            ]
        else:
            port_offset = max(-min(source.h, target.h) * 0.22, min(min(source.h, target.h) * 0.22, route_offset * 0.3))
            overlap_top = max(source.top, target.top)
            overlap_bottom = min(source.bottom, target.bottom)
            shared_y = (overlap_top + overlap_bottom) / 2 + port_offset
            shared_y = max(overlap_top + 2, min(overlap_bottom - 2, shared_y))
            points_list = [(start_x, shared_y), (end_x, shared_y)]
    else:
        port_limit = min(source.w, target.w) * 0.30
        direction_bias = -8 if target.top >= source.bottom else 8
        port_offset = max(-port_limit, min(port_limit, route_offset + direction_bias))
        start_x = source.cx + port_offset
        end_x = target.cx + port_offset
        lane_distance = max(16, min(64, 32 + route_offset * 0.55))
        if target.top >= source.bottom:
            start_y = source.bottom
            end_y = target.top
            lane_y = start_y + lane_distance
        else:
            start_y = source.top
            end_y = target.bottom
            lane_y = start_y - lane_distance
        points_list = [(start_x, start_y), (start_x, lane_y), (end_x, lane_y), (end_x, end_y)]
    simplified_points: list[tuple[float, float]] = []
    for point in points_list:
        if simplified_points and math.dist(simplified_points[-1], point) <= 0.5:
            continue
        if len(simplified_points) >= 2:
            first = simplified_points[-2]
            second = simplified_points[-1]
            same_x = abs(first[0] - second[0]) <= 0.5 and abs(second[0] - point[0]) <= 0.5
            same_y = abs(first[1] - second[1]) <= 0.5 and abs(second[1] - point[1]) <= 0.5
            if same_x or same_y:
                simplified_points[-1] = point
                continue
        simplified_points.append(point)
    return RoutedEdge(source, target, color, dash, str(label), simplified_points)


def route_segment_boxes(points: list[tuple[float, float]]) -> list[Box]:
    boxes: list[Box] = []
    for index, (first, second) in enumerate(zip(points, points[1:])):
        x1, y1 = first
        x2, y2 = second
        x = min(x1, x2)
        y = min(y1, y2)
        width = abs(x2 - x1)
        height = abs(y2 - y1)
        if width <= 0.5:
            x -= 2
            width = 4
        elif height <= 0.5:
            y -= 2
            height = 4
        else:
            x -= 2
            y -= 2
            width += 4
            height += 4
        boxes.append(Box(f"route-segment-{index}", x, y, width, height))
    return boxes


def route_segment_crossing(
    a1: tuple[float, float],
    a2: tuple[float, float],
    b1: tuple[float, float],
    b2: tuple[float, float],
    endpoint_clearance: float = 2.0,
) -> tuple[float, float] | None:
    a_dx, a_dy = a2[0] - a1[0], a2[1] - a1[1]
    b_dx, b_dy = b2[0] - b1[0], b2[1] - b1[1]
    denominator = a_dx * b_dy - a_dy * b_dx
    if abs(denominator) <= 0.5:
        return None
    offset_x, offset_y = b1[0] - a1[0], b1[1] - a1[1]
    a_ratio = (offset_x * b_dy - offset_y * b_dx) / denominator
    b_ratio = (offset_x * a_dy - offset_y * a_dx) / denominator
    a_margin = min(0.49, endpoint_clearance / max(math.hypot(a_dx, a_dy), endpoint_clearance * 2))
    b_margin = min(0.49, endpoint_clearance / max(math.hypot(b_dx, b_dy), endpoint_clearance * 2))
    if not (a_margin < a_ratio < 1 - a_margin and b_margin < b_ratio < 1 - b_margin):
        return None
    return a1[0] + a_ratio * a_dx, a1[1] + a_ratio * a_dy


def edge_svg(
    route: RoutedEdge,
    boxes: dict[str, Box],
    placed_label_boxes: list[Box],
    other_route_boxes: list[Box],
    canvas_w: float,
    bridge_points: list[tuple[float, float]],
) -> str:
    source = route.source
    target = route.target
    points_list = route.points
    points = " ".join(f"{x:.1f},{y:.1f}" for x, y in points_list)
    label = route.label
    label_lines = wrap_text(label, 28)[:2] if label else []
    label_width = max(96, min(max((weighted_text_width(line) for line in label_lines), default=0) + 24, 220))
    label_height = 24 if len(label_lines) <= 1 else 42
    label_svg = ""
    if label:
        horizontal_segments = [
            (first, second)
            for first, second in zip(points_list, points_list[1:])
            if abs(first[1] - second[1]) <= 0.5 and abs(first[0] - second[0]) >= 2
        ]
        horizontal_segments.sort(key=lambda segment: abs(segment[1][0] - segment[0][0]), reverse=True)
        candidates: list[Box] = []
        for first, second in horizontal_segments:
            for fraction in (0.5, 0.33, 0.67, 0.2, 0.8):
                center_x = first[0] + (second[0] - first[0]) * fraction
                for vertical_offset in (
                    -label_height - 2,
                    4,
                    -2 * label_height - 12,
                    label_height + 12,
                    -3 * label_height - 22,
                    2 * label_height + 22,
                ):
                    y = first[1] + vertical_offset
                    candidates.append(Box("edge-label", center_x - label_width / 2, y, label_width, label_height))
        vertical_segments = [
            (first, second)
            for first, second in zip(points_list, points_list[1:])
            if abs(first[0] - second[0]) <= 0.5 and abs(first[1] - second[1]) >= 40
        ]
        vertical_segments.sort(key=lambda segment: abs(segment[1][1] - segment[0][1]), reverse=True)
        for first, second in vertical_segments:
            for fraction in (0.5, 0.33, 0.67, 0.2, 0.8):
                center_y = first[1] + (second[1] - first[1]) * fraction
                for horizontal_gap in (10, 24, 40):
                    candidates.append(
                        Box(
                            "edge-label",
                            first[0] - label_width - horizontal_gap,
                            center_y - label_height / 2,
                            label_width,
                            label_height,
                        )
                    )
                    candidates.append(
                        Box("edge-label", first[0] + horizontal_gap, center_y - label_height / 2, label_width, label_height)
                    )
        if not candidates:
            first, second = points_list[0], points_list[-1]
            candidates.append(
                Box(
                    "edge-label",
                    (first[0] + second[0]) / 2 - label_width / 2,
                    (first[1] + second[1]) / 2 - label_height - 2,
                    label_width,
                    label_height,
                )
            )
        candidates = [
            candidate
            for candidate in candidates
            if candidate.x >= 20 and candidate.right <= canvas_w - 20 and candidate.y >= 110
        ] or candidates

        def overlaps(first: Box, second: Box, padding: float = 4) -> bool:
            return not (
                first.right + padding <= second.x
                or second.right + padding <= first.x
                or first.bottom + padding <= second.y
                or second.bottom + padding <= first.y
            )

        obstacles = list(boxes.values()) + placed_label_boxes + other_route_boxes
        label_box = next(
            (candidate for candidate in candidates if not any(overlaps(candidate, obstacle) for obstacle in obstacles)),
            candidates[0],
        )
        placed_label_boxes.append(label_box)
        label_svg = (
            f'<rect x="{label_box.x:.1f}" y="{label_box.y:.1f}" width="{label_width:.1f}" '
            f'height="{label_height:.1f}" rx="12" fill="white" stroke="#e5e7eb"/>'
            f'{svg_text(label_lines, label_box.x+12, label_box.y+16, 11, "600", "#475569")}'
        )
    bridge_svg = "".join(
        f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5.5" fill="#ffffff" stroke="#ffffff" stroke-width="2"/>'
        for x, y in bridge_points
    )
    bridge_attr = ' data-crossing-style="bridge"' if bridge_points else ""
    return (
        f'<g class="edge edge-{esc(source.id)}-{esc(target.id)}"{bridge_attr}>'
        f"{bridge_svg}"
        f'<polyline points="{points}" fill="none" stroke="{route.color}" stroke-width="2.5" '
        f'marker-end="url(#arrow)"{route.dash}/>'
        f"{label_svg}</g>"
    )


def group_svg(group: dict[str, Any], boxes: dict[str, Box]) -> str:
    node_ids = [str(x) for x in group.get("nodes", [])]
    group_boxes = [boxes[x] for x in node_ids if x in boxes]
    if not group_boxes:
        return ""

    x1 = min(b.x for b in group_boxes) - GROUP_PAD_X
    y1 = min(b.y for b in group_boxes) - GROUP_PAD_Y
    x2 = max(b.x + b.w for b in group_boxes) + GROUP_PAD_X
    y2 = max(b.y + b.h for b in group_boxes) + GROUP_PAD_Y
    fill = group.get("fill", "#f8fafc")
    stroke = group.get("stroke", "#cbd5e1")
    label = str(group.get("label", group.get("id", "")))
    dash = ' stroke-dasharray="8 8"' if group.get("style") == "dashed" else ""
    return (
        f'<g id="group-{esc(group.get("id", ""))}">'
        f'<rect x="{x1:.1f}" y="{y1:.1f}" width="{x2-x1:.1f}" height="{y2-y1:.1f}" '
        f'rx="26" fill="{fill}" fill-opacity="0.38" stroke="{stroke}" stroke-width="2"{dash}/>'
        f'{svg_text([label], x1 + 22, y1 + 30, 16, "800", stroke)}'
        f"</g>"
    )


def callout_box(
    callout: dict[str, Any],
    boxes: dict[str, Box],
    index: int,
    canvas_w: float,
    groups: list[dict[str, Any]],
) -> Box:
    layout = callout.get("layout") or {}
    label = str(callout.get("label") or callout.get("id") or "callout")
    w = float(layout.get("w") or min(max(weighted_text_width(label) + 32, 130), 420))
    h = float(layout.get("h") or CALLOUT_H)
    x = layout.get("x")
    y = layout.get("y")

    target_id = str(callout.get("target") or "")
    if not target_id and callout.get("nodes"):
        target_id = str(callout.get("nodes", [None])[0])
    target = boxes.get(target_id)

    lines = wrap_text(label, max(12, int((w - 24) / 8)))[:2]
    if len(lines) > 1:
        h = max(h, 54)

    if target and x is None and y is None:
        group_bounds: Box | None = None
        for group in groups:
            member_ids = {str(node_id) for node_id in group.get("nodes", [])}
            if target.id not in member_ids:
                continue
            members = [boxes[node_id] for node_id in member_ids if node_id in boxes]
            if members:
                left = min(box.x for box in members) - GROUP_PAD_X
                top = min(box.y for box in members) - GROUP_PAD_Y
                right = max(box.right for box in members) + GROUP_PAD_X
                bottom = max(box.bottom for box in members) + GROUP_PAD_Y
                group_bounds = Box("callout-group", left, top, right - left, bottom - top)
            break

        candidates = [
            Box("callout", target.cx - w / 2, target.bottom + 14, w, h),
            Box("callout", target.right + 18, target.cy - h / 2, w, h),
            Box("callout", target.x - w - 18, target.cy - h / 2, w, h),
            Box("callout", target.cx - w / 2, target.y - h - 14, w, h),
        ]

        def candidate_fits(candidate: Box) -> bool:
            if candidate.x < 20 or candidate.right > canvas_w - 20 or candidate.y < 110:
                return False
            if group_bounds and (
                candidate.x < group_bounds.x + 12
                or candidate.right > group_bounds.right - 12
                or candidate.y < group_bounds.y + 12
                or candidate.bottom > group_bounds.bottom - 12
            ):
                return False
            return not any(
                candidate.x < box.right + 8
                and candidate.right > box.x - 8
                and candidate.y < box.bottom + 8
                and candidate.bottom > box.y - 8
                for box in boxes.values()
            )

        chosen = next((candidate for candidate in candidates if candidate_fits(candidate)), candidates[0])
        x, y = chosen.x, chosen.y
    else:
        if x is None:
            x = target.cx - w / 2 if target else MARGIN_X + (index % 3) * 440
        if y is None:
            y = target.bottom + 14 if target else TOP_Y + index * (h + 12)
    return Box(str(callout.get("id") or f"callout_{index}"), float(x), float(y), w, h)


def callout_svg(callout: dict[str, Any], box: Box) -> str:
    label = str(callout.get("label") or callout.get("id") or "callout")
    fill = str(callout.get("fill") or "#ffffff")
    stroke = str(callout.get("stroke") or "#cbd5e1")
    dash = ' stroke-dasharray="7 6"' if callout.get("style") == "dashed" else ""
    lines = wrap_text(label, max(12, int((box.w - 24) / 8)))[:2]
    return (
        f'<g id="callout-{esc(box.id)}">'
        f'<rect x="{box.x:.1f}" y="{box.y:.1f}" width="{box.w:.1f}" height="{box.h:.1f}" '
        f'rx="{box.h / 2:.1f}" fill="{fill}" stroke="{stroke}" stroke-width="1.4"{dash}/>'
        f'{svg_text(lines, box.x + 14, box.y + 22, 12, "700", "#475569")}'
        f"</g>"
    )


def render_svg(schema: dict[str, Any]) -> str:
    meta = schema.get("meta", {})
    categories = schema.get("categories", {})
    nodes = schema.get("nodes", [])
    edges = schema.get("edges", [])
    groups = schema.get("groups", [])
    callouts = schema.get("callouts", [])
    canvas_w = float(meta.get("canvas_width") or default_canvas_width(nodes))
    boxes = compute_layout(nodes, edges, canvas_w, groups, schema.get("composition") or {})
    nodes_by_id = {str(n["id"]): n for n in nodes}
    callout_boxes = [callout_box(callout, boxes, index, canvas_w, groups) for index, callout in enumerate(callouts)]
    edge_groups: dict[tuple[str, str], list[int]] = {}
    for index, edge in enumerate(edges):
        source_node = nodes_by_id.get(str(edge.get("from")), {})
        target_node = nodes_by_id.get(str(edge.get("to")), {})
        edge_groups.setdefault((layer_key(source_node), layer_key(target_node)), []).append(index)
    edge_offsets: dict[int, float] = {}
    for indexes in edge_groups.values():
        for position, edge_index in enumerate(indexes):
            edge_offsets[edge_index] = (position - (len(indexes) - 1) / 2) * 32
    routes = [
        route
        for index, edge in enumerate(edges)
        if (route := route_edge(edge, boxes, categories, nodes_by_id, edge_offsets.get(index, 0))) is not None
    ]
    segment_boxes = [route_segment_boxes(route.points) for route in routes]
    bridge_points_by_route: list[list[tuple[float, float]]] = [[] for _ in routes]
    for first_index, first_route in enumerate(routes):
        first_segments = list(zip(first_route.points, first_route.points[1:]))
        for second_index in range(first_index + 1, len(routes)):
            second_route = routes[second_index]
            second_segments = list(zip(second_route.points, second_route.points[1:]))
            for first_segment in first_segments:
                for second_segment in second_segments:
                    crossing = route_segment_crossing(*first_segment, *second_segment)
                    if crossing and not any(math.dist(crossing, point) <= 1 for point in bridge_points_by_route[second_index]):
                        bridge_points_by_route[second_index].append(crossing)
    placed_label_boxes: list[Box] = []
    edge_svgs: list[str] = []
    for index, route in enumerate(routes):
        other_route_boxes = [
            box
            for route_index, boxes_for_route in enumerate(segment_boxes)
            if route_index != index
            for box in boxes_for_route
        ]
        edge_svgs.append(
            edge_svg(route, boxes, placed_label_boxes, other_route_boxes, canvas_w, bridge_points_by_route[index])
        )

    route_y = [y for route in routes for _, y in route.points]
    route_x = [x for route in routes for x, _ in route.points]
    content_bottom = max(
        [b.bottom for b in boxes.values()]
        + [b.bottom for b in callout_boxes]
        + [b.bottom for b in placed_label_boxes]
        + route_y,
        default=TOP_Y + CARD_H,
    )
    group_bottom = max(
        (
            max((boxes[str(node_id)].bottom for node_id in group.get("nodes", []) if str(node_id) in boxes), default=0)
            + GROUP_PAD_Y
            for group in groups
        ),
        default=0,
    )
    group_right = max(
        (
            max((boxes[str(node_id)].right for node_id in group.get("nodes", []) if str(node_id) in boxes), default=0)
            + GROUP_PAD_X
            for group in groups
        ),
        default=0,
    )
    content_right = max(
        [b.right for b in boxes.values()]
        + [b.right for b in callout_boxes]
        + [b.right for b in placed_label_boxes]
        + route_x
        + [group_right],
        default=canvas_w,
    )
    canvas_w = math.ceil(max(canvas_w, content_right + 42))
    canvas_h = math.ceil(max(content_bottom, group_bottom) + 42)

    parts = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{int(canvas_w)}" height="{canvas_h}" viewBox="0 0 {int(canvas_w)} {canvas_h}">',
        "<defs>",
        '<marker id="arrow" markerWidth="12" markerHeight="12" refX="10" refY="6" orient="auto" markerUnits="strokeWidth">',
        '<path d="M2,2 L10,6 L2,10 Z" fill="#64748b"/>',
        "</marker>",
        "</defs>",
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        svg_text(
            [str(meta.get("title", "Architecture Diagram"))], canvas_w / 2, 62, 34, "800", "#0f172a", "middle"
        ),
    ]
    subtitle = meta.get("subtitle")
    if subtitle:
        parts.append(svg_text([str(subtitle)], canvas_w / 2, 96, 17, "400", "#64748b", "middle"))
    updated = meta.get("updated")
    if updated:
        parts.append(svg_text([f"source: schema.yaml · {updated}"], canvas_w - 24, 20, 12, "600", "#94a3b8", "end"))

    if groups:
        parts.append('<g id="groups">')
        for group in groups:
            parts.append(group_svg(group, boxes))
        parts.append("</g>")

    parts.append('<g id="edges">')
    parts.extend(edge_svgs)
    parts.append("</g>")

    parts.append('<g id="nodes">')
    for node in nodes:
        parts.append(node_svg(node, boxes[str(node["id"])], categories))
    parts.append("</g>")

    if callouts:
        parts.append('<g id="callouts">')
        for callout, box in zip(callouts, callout_boxes):
            parts.append(callout_svg(callout, box))
        parts.append("</g>")
    parts.append("</svg>")
    return "\n".join(parts)


def write_html(svg: str, path: pathlib.Path) -> None:
    path.write_text(
        "<!doctype html><meta charset='utf-8'><title>Architecture Diagram</title>"
        "<style>body{margin:0;background:#f8fafc}svg{display:block;max-width:100%;height:auto;margin:auto}</style>"
        + svg,
        encoding="utf-8",
    )


def maybe_write_png(svg_path: pathlib.Path, png_path: pathlib.Path) -> bool:
    try:
        import cairosvg  # type: ignore
    except (ImportError, OSError):
        return False
    cairosvg.svg2png(url=str(svg_path), write_to=str(png_path), output_width=CANVAS_W)
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--schema", default="schema.yaml")
    parser.add_argument("--out-dir", default=".")
    parser.add_argument("--png", action="store_true", help="also render a PNG if cairosvg is installed")
    parser.add_argument(
        "--png-name",
        default="diagram.png",
        help="PNG filename inside --out-dir (default: diagram.png)",
    )
    args = parser.parse_args()

    schema_path = pathlib.Path(args.schema)
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    schema = load_schema(schema_path)
    svg = render_svg(schema)

    svg_path = out_dir / "diagram.svg"
    html_path = out_dir / "diagram.html"
    png_path = out_dir / args.png_name
    svg_path.write_text(svg, encoding="utf-8")
    write_html(svg, html_path)
    rendered_png = maybe_write_png(svg_path, png_path) if args.png else False

    print(f"wrote {svg_path}")
    print(f"wrote {html_path}")
    if args.png:
        if rendered_png:
            print(f"wrote {png_path}")
        else:
            print(f"skipped {png_path.name}: install cairosvg or run with `uv run --with cairosvg`")


if __name__ == "__main__":
    main()

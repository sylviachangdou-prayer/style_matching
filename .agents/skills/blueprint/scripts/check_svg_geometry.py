#!/usr/bin/env python3
"""Browser-measured geometry checks for Blueprint SVG render experiments."""

from __future__ import annotations

import argparse
import json
import math
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

from playwright.sync_api import Error as PlaywrightError
from playwright.sync_api import sync_playwright


MEASURE_JS = r"""
(svg) => {
  const box = (element) => {
    if (!element) return null;
    const b = element.getBBox();
    return {x: b.x, y: b.y, width: b.width, height: b.height,
            right: b.x + b.width, bottom: b.y + b.height};
  };
  const union = (boxes) => {
    const usable = boxes.filter(Boolean);
    if (!usable.length) return null;
    const x = Math.min(...usable.map((b) => b.x));
    const y = Math.min(...usable.map((b) => b.y));
    const right = Math.max(...usable.map((b) => b.right));
    const bottom = Math.max(...usable.map((b) => b.bottom));
    return {x, y, width: right - x, height: bottom - y, right, bottom};
  };
  const textRecord = (element, index) => element ? ({
    index,
    text: (element.textContent || '').replace(/\s+/g, ' ').trim(),
    bbox: box(element),
  }) : null;

  const nodes = [...svg.querySelectorAll('g[id^="node-"]')].map((element) => {
    const children = [...element.children];
    const primary = children.find((child) => {
      const tag = child.tagName.toLowerCase();
      return tag === 'rect' || tag === 'path';
    });
    const decorations = children
      .filter((child) => {
        const tag = child.tagName.toLowerCase();
        return tag === 'line' || tag === 'ellipse' || (tag === 'path' && child !== primary);
      })
      .map((child, index) => ({index, tag: child.tagName.toLowerCase(), bbox: box(child)}));
    return {
      id: element.id.replace(/^node-/, ''),
      bbox: box(element),
      frame: box(primary),
      texts: [...element.querySelectorAll('text')].map(textRecord),
      decorations,
    };
  });

  const edges = [...svg.querySelectorAll('g.edge')].map((element, index) => {
    const polyline = element.querySelector('polyline');
    const points = [];
    if (polyline) {
      for (let i = 0; i < polyline.points.numberOfItems; i += 1) {
        const point = polyline.points.getItem(i);
        points.push({x: point.x, y: point.y});
      }
    }
    const classId = [...element.classList].find((name) => name.startsWith('edge-'));
    return {
      id: classId || `edge-${index}`,
      bbox: box(element),
      points,
      labelRect: box(element.querySelector(':scope > rect')),
      labelText: textRecord(element.querySelector(':scope > text'), 0),
      crossingStyle: element.dataset.crossingStyle || '',
    };
  });

  const groups = [...svg.querySelectorAll('g[id^="group-"]')].map((element) => ({
    id: element.id.replace(/^group-/, ''),
    frame: box(element.querySelector(':scope > rect')),
  }));
  const callouts = [...svg.querySelectorAll('g[id^="callout-"]')].map((element) => ({
    id: element.id.replace(/^callout-/, ''),
    frame: box(element.querySelector(':scope > rect')),
  }));
  const headers = [...svg.querySelectorAll(':scope > text')].map(textRecord);
  const contentBoxes = [...svg.children]
    .filter((element) => {
      const tag = element.tagName.toLowerCase();
      if (tag === 'defs') return false;
      return !(tag === 'rect' && element.getAttribute('width') === '100%');
    })
    .map(box);
  const architectureBoxes = [...svg.querySelectorAll(':scope > g#groups, :scope > g#edges, :scope > g#nodes, :scope > g#callouts')]
    .map(box);
  const viewBox = svg.viewBox.baseVal;

  return {
    viewBox: {x: viewBox.x, y: viewBox.y, width: viewBox.width, height: viewBox.height,
              right: viewBox.x + viewBox.width, bottom: viewBox.y + viewBox.height},
    content: union(contentBoxes),
    architecture: union(architectureBoxes),
    nodes,
    edges,
    groups,
    callouts,
    headers,
  };
}
"""


def intersection(a: dict[str, float], b: dict[str, float], tolerance: float = 0.0) -> dict[str, float] | None:
    x = max(a["x"], b["x"])
    y = max(a["y"], b["y"])
    right = min(a["right"], b["right"])
    bottom = min(a["bottom"], b["bottom"])
    if right - x <= tolerance or bottom - y <= tolerance:
        return None
    return make_box(x, y, right - x, bottom - y)


def make_box(x: float, y: float, width: float, height: float) -> dict[str, float]:
    return {
        "x": round(x, 3),
        "y": round(y, 3),
        "width": round(width, 3),
        "height": round(height, 3),
        "right": round(x + width, 3),
        "bottom": round(y + height, 3),
    }


def inflate(box: dict[str, float], amount: float) -> dict[str, float]:
    return make_box(box["x"] - amount, box["y"] - amount, box["width"] + 2 * amount, box["height"] + 2 * amount)


def contains(outer: dict[str, float], inner: dict[str, float], tolerance: float = 1.0) -> bool:
    return (
        inner["x"] >= outer["x"] - tolerance
        and inner["y"] >= outer["y"] - tolerance
        and inner["right"] <= outer["right"] + tolerance
        and inner["bottom"] <= outer["bottom"] + tolerance
    )


def segment_overlap(
    a1: dict[str, float], a2: dict[str, float], b1: dict[str, float], b2: dict[str, float], epsilon: float = 0.5
) -> dict[str, float] | None:
    a_horizontal = abs(a1["y"] - a2["y"]) <= epsilon
    b_horizontal = abs(b1["y"] - b2["y"]) <= epsilon
    a_vertical = abs(a1["x"] - a2["x"]) <= epsilon
    b_vertical = abs(b1["x"] - b2["x"]) <= epsilon

    if a_horizontal and b_horizontal and abs(a1["y"] - b1["y"]) <= epsilon:
        start = max(min(a1["x"], a2["x"]), min(b1["x"], b2["x"]))
        end = min(max(a1["x"], a2["x"]), max(b1["x"], b2["x"]))
        if end - start > 2.0:
            return make_box(start, a1["y"] - 2, end - start, 4)
    if a_vertical and b_vertical and abs(a1["x"] - b1["x"]) <= epsilon:
        start = max(min(a1["y"], a2["y"]), min(b1["y"], b2["y"]))
        end = min(max(a1["y"], a2["y"]), max(b1["y"], b2["y"]))
        if end - start > 2.0:
            return make_box(a1["x"] - 2, start, 4, end - start)
    return None


def segment_crossing(
    a1: dict[str, float],
    a2: dict[str, float],
    b1: dict[str, float],
    b2: dict[str, float],
    epsilon: float = 0.5,
    endpoint_clearance: float = 2.0,
) -> dict[str, float] | None:
    a_dx = a2["x"] - a1["x"]
    a_dy = a2["y"] - a1["y"]
    b_dx = b2["x"] - b1["x"]
    b_dy = b2["y"] - b1["y"]
    denominator = a_dx * b_dy - a_dy * b_dx
    if abs(denominator) <= epsilon:
        return None

    offset_x = b1["x"] - a1["x"]
    offset_y = b1["y"] - a1["y"]
    a_ratio = (offset_x * b_dy - offset_y * b_dx) / denominator
    b_ratio = (offset_x * a_dy - offset_y * a_dx) / denominator
    a_length = math.hypot(a_dx, a_dy)
    b_length = math.hypot(b_dx, b_dy)
    a_margin = min(0.49, endpoint_clearance / max(a_length, endpoint_clearance * 2))
    b_margin = min(0.49, endpoint_clearance / max(b_length, endpoint_clearance * 2))
    if not (a_margin < a_ratio < 1 - a_margin and b_margin < b_ratio < 1 - b_margin):
        return None

    x = a1["x"] + a_ratio * a_dx
    y = a1["y"] + a_ratio * a_dy
    return make_box(x - 4, y - 4, 8, 8)


def segment_box(
    first: dict[str, float], second: dict[str, float], padding: float = 2.0
) -> dict[str, float]:
    x = min(first["x"], second["x"]) - padding
    y = min(first["y"], second["y"]) - padding
    width = abs(second["x"] - first["x"]) + 2 * padding
    height = abs(second["y"] - first["y"]) + 2 * padding
    return make_box(x, y, max(width, 2 * padding), max(height, 2 * padding))


def segment_inside_box(
    first: dict[str, float], second: dict[str, float], box: dict[str, float], epsilon: float = 1.0
) -> dict[str, float] | None:
    horizontal = abs(first["y"] - second["y"]) <= epsilon
    vertical = abs(first["x"] - second["x"]) <= epsilon
    if horizontal and box["y"] + epsilon < first["y"] < box["bottom"] - epsilon:
        start = max(min(first["x"], second["x"]), box["x"] + epsilon)
        end = min(max(first["x"], second["x"]), box["right"] - epsilon)
        if end - start > 2.0:
            return make_box(start, first["y"] - 2, end - start, 4)
    if vertical and box["x"] + epsilon < first["x"] < box["right"] - epsilon:
        start = max(min(first["y"], second["y"]), box["y"] + epsilon)
        end = min(max(first["y"], second["y"]), box["bottom"] - epsilon)
        if end - start > 2.0:
            return make_box(first["x"] - 2, start, 4, end - start)
    return None


def union_boxes(boxes: list[dict[str, float]]) -> dict[str, float]:
    x = min(box["x"] for box in boxes)
    y = min(box["y"] for box in boxes)
    right = max(box["right"] for box in boxes)
    bottom = max(box["bottom"] for box in boxes)
    return make_box(x, y, right - x, bottom - y)


def resolve_edge_endpoints(edge_id: str, node_ids: set[str]) -> tuple[str | None, str | None]:
    suffix = edge_id.removeprefix("edge-")
    for source in sorted(node_ids, key=len, reverse=True):
        prefix = source + "-"
        if suffix.startswith(prefix) and suffix[len(prefix) :] in node_ids:
            return source, suffix[len(prefix) :]
    return None, None


def add_issue(
    issues: list[dict[str, Any]], severity: str, kind: str, message: str, bbox: dict[str, float] | None = None, **details: Any
) -> None:
    issue: dict[str, Any] = {"severity": severity, "kind": kind, "message": message}
    if bbox:
        issue["bbox"] = bbox
    issue.update(details)
    issues.append(issue)


def launch_browser(playwright: Any) -> tuple[Any, str]:
    attempts = [({}, "bundled chromium"), ({"channel": "msedge"}, "Microsoft Edge"), ({"channel": "chrome"}, "Google Chrome")]
    errors: list[str] = []
    for options, label in attempts:
        try:
            return playwright.chromium.launch(headless=True, **options), label
        except PlaywrightError as exc:
            errors.append(f"{label}: {exc}")
    raise RuntimeError("Could not launch a Chromium browser:\n" + "\n".join(errors))


def measure_svg(svg_path: Path, screenshot_out: Path | None) -> tuple[dict[str, Any], str]:
    with sync_playwright() as playwright:
        browser, browser_label = launch_browser(playwright)
        page = browser.new_page(viewport={"width": 1920, "height": 1080}, device_scale_factor=1)
        page.goto(svg_path.resolve().as_uri(), wait_until="load")
        svg = page.locator("svg")
        if svg.count() != 1:
            browser.close()
            raise RuntimeError(f"Expected one SVG root, found {svg.count()}")
        measured = svg.evaluate(MEASURE_JS)
        view_box = measured["viewBox"]
        page.set_viewport_size({"width": math.ceil(view_box["width"]), "height": math.ceil(view_box["height"])})
        if screenshot_out:
            screenshot_out.parent.mkdir(parents=True, exist_ok=True)
            page.screenshot(
                path=str(screenshot_out),
                clip={"x": 0, "y": 0, "width": view_box["width"], "height": view_box["height"]},
            )
        browser.close()
    return measured, browser_label


def analyze(
    measured: dict[str, Any], max_margin_ratio: float, min_occupancy: float, min_group_gap: float
) -> dict[str, Any]:
    issues: list[dict[str, Any]] = []
    nodes = measured["nodes"]
    edges = measured["edges"]
    node_ids = {node["id"] for node in nodes}
    for edge in edges:
        edge["source"], edge["target"] = resolve_edge_endpoints(edge["id"], node_ids)

    for node in nodes:
        frame = node["frame"]
        if not frame:
            add_issue(issues, "error", "missing_node_frame", f"Node {node['id']} has no measurable frame", node=node["id"])
            continue
        for text in node["texts"]:
            if text["bbox"] and not contains(frame, text["bbox"]):
                add_issue(
                    issues,
                    "error",
                    "text_outside_node",
                    f"Text escapes node {node['id']}: {text['text']}",
                    text["bbox"],
                    node=node["id"],
                    text=text["text"],
                    frame=frame,
                )
        for index, first in enumerate(node["texts"]):
            for second in node["texts"][index + 1 :]:
                if not first["bbox"] or not second["bbox"]:
                    continue
                overlap = intersection(first["bbox"], second["bbox"], tolerance=1.0)
                if overlap:
                    add_issue(
                        issues,
                        "error",
                        "text_overlap",
                        f"Text overlaps inside node {node['id']}: {first['text']} / {second['text']}",
                        overlap,
                        node=node["id"],
                        texts=[first["text"], second["text"]],
                    )
        for decoration in node["decorations"]:
            if not decoration["bbox"]:
                continue
            decoration_box = inflate(decoration["bbox"], 1.5)
            for text in node["texts"]:
                if not text["bbox"]:
                    continue
                overlap = intersection(decoration_box, text["bbox"], tolerance=1.0)
                if overlap:
                    add_issue(
                        issues,
                        "error",
                        "decoration_text_collision",
                        f"{decoration['tag']} crosses text in node {node['id']}: {text['text']}",
                        overlap,
                        node=node["id"],
                        text=text["text"],
                        decoration=decoration["tag"],
                    )

    for index, first in enumerate(nodes):
        if not first["frame"]:
            continue
        for second in nodes[index + 1 :]:
            if not second["frame"]:
                continue
            overlap = intersection(first["frame"], second["frame"], tolerance=1.0)
            if overlap:
                add_issue(
                    issues,
                    "error",
                    "node_collision",
                    f"Nodes overlap: {first['id']} / {second['id']}",
                    overlap,
                    nodes=[first["id"], second["id"]],
                )

    group_gaps: list[dict[str, Any]] = []
    for index, first in enumerate(measured["groups"]):
        if not first["frame"]:
            continue
        for second in measured["groups"][index + 1 :]:
            if not second["frame"]:
                continue
            overlap = intersection(first["frame"], second["frame"], tolerance=0.5)
            if overlap:
                add_issue(
                    issues,
                    "error",
                    "group_collision",
                    f"Groups overlap: {first['id']} / {second['id']}",
                    overlap,
                    groups=[first["id"], second["id"]],
                )
                continue

            x_overlap = min(first["frame"]["right"], second["frame"]["right"]) - max(
                first["frame"]["x"], second["frame"]["x"]
            )
            y_overlap = min(first["frame"]["bottom"], second["frame"]["bottom"]) - max(
                first["frame"]["y"], second["frame"]["y"]
            )
            axis = None
            gap = None
            if x_overlap > 0:
                axis = "vertical"
                gap = max(
                    second["frame"]["y"] - first["frame"]["bottom"],
                    first["frame"]["y"] - second["frame"]["bottom"],
                )
            elif y_overlap > 0:
                axis = "horizontal"
                gap = max(
                    second["frame"]["x"] - first["frame"]["right"],
                    first["frame"]["x"] - second["frame"]["right"],
                )
            if axis is None or gap is None or gap < 0:
                continue
            gap_record = {
                "groups": [first["id"], second["id"]],
                "axis": axis,
                "gap": round(gap, 3),
            }
            group_gaps.append(gap_record)
            if gap < min_group_gap:
                add_issue(
                    issues,
                    "warning",
                    "tight_group_spacing",
                    f"Groups are only {gap:.1f}px apart: {first['id']} / {second['id']}",
                    groups=gap_record["groups"],
                    axis=axis,
                    gap=round(gap, 3),
                    minimum=round(min_group_gap, 3),
                )

    for index, first in enumerate(edges):
        endpoints = {first.get("source"), first.get("target")}
        segments = list(zip(first["points"], first["points"][1:]))
        non_orthogonal = [
            segment_box(*segment)
            for segment in segments
            if abs(segment[0]["x"] - segment[1]["x"]) > 0.5
            and abs(segment[0]["y"] - segment[1]["y"]) > 0.5
        ]
        if non_orthogonal:
            add_issue(
                issues,
                "error",
                "non_orthogonal_edge",
                f"Edge {first['id']} contains a diagonal route segment",
                union_boxes(non_orthogonal),
                edge=first["id"],
            )
        for node in nodes:
            if not node["frame"]:
                continue
            if first["labelRect"]:
                overlap = intersection(first["labelRect"], node["frame"], tolerance=1.0)
                if overlap:
                    add_issue(
                        issues,
                        "error",
                        "edge_label_node_collision",
                        f"Edge label {first['id']} overlaps node {node['id']}",
                        overlap,
                        edge=first["id"],
                        node=node["id"],
                    )
            if node["id"] in endpoints:
                continue
            route_overlaps = [
                overlap
                for segment in segments
                if (overlap := segment_inside_box(*segment, node["frame"])) is not None
            ]
            if route_overlaps:
                add_issue(
                    issues,
                    "error",
                    "edge_crosses_node",
                    f"Edge {first['id']} crosses non-endpoint node {node['id']}",
                    union_boxes(route_overlaps),
                    edge=first["id"],
                    node=node["id"],
                )
        for second in edges[index + 1 :]:
            if first["labelRect"] and second["labelRect"]:
                overlap = intersection(first["labelRect"], second["labelRect"], tolerance=1.0)
                if overlap:
                    add_issue(
                        issues,
                        "error",
                        "edge_label_collision",
                        f"Edge labels overlap: {first['id']} / {second['id']}",
                        overlap,
                        edges=[first["id"], second["id"]],
                    )
            first_segments = segments
            second_segments = list(zip(second["points"], second["points"][1:]))
            for label_edge, route_edge, route_segments in (
                (first, second, second_segments),
                (second, first, first_segments),
            ):
                if not label_edge["labelRect"]:
                    continue
                label_route_overlaps = [
                    overlap
                    for segment in route_segments
                    if (overlap := segment_inside_box(*segment, label_edge["labelRect"], epsilon=0.5)) is not None
                ]
                if label_route_overlaps:
                    add_issue(
                        issues,
                        "error",
                        "edge_route_label_collision",
                        f"Edge route {route_edge['id']} crosses label {label_edge['id']}",
                        union_boxes(label_route_overlaps),
                        route_edge=route_edge["id"],
                        label_edge=label_edge["id"],
                    )
            route_overlaps: list[dict[str, float]] = []
            route_crossings: list[dict[str, float]] = []
            for first_segment in first_segments:
                for second_segment in second_segments:
                    overlap = segment_overlap(*first_segment, *second_segment)
                    if overlap:
                        route_overlaps.append(overlap)
                    crossing = segment_crossing(*first_segment, *second_segment)
                    if crossing:
                        route_crossings.append(crossing)
            if route_overlaps:
                shared_endpoint = (
                    first.get("source") == second.get("source")
                    or first.get("target") == second.get("target")
                )
                kind = "shared_edge_trunk" if shared_endpoint else "edge_segment_overlap"
                severity = "warning" if shared_endpoint else "error"
                message = "Edges share a route trunk" if shared_endpoint else "Unrelated or opposing edge routes overlap"
                add_issue(
                    issues,
                    severity,
                    kind,
                    f"{message}: {first['id']} / {second['id']}",
                    union_boxes(route_overlaps),
                    edges=[first["id"], second["id"]],
                )
            if route_crossings:
                bridged = first.get("crossingStyle") == "bridge" or second.get("crossingStyle") == "bridge"
                add_issue(
                    issues,
                    "warning" if bridged else "error",
                    "bridged_edge_crossing" if bridged else "edge_route_crossing",
                    (
                        f"Edge routes use an explicit bridge: {first['id']} / {second['id']}"
                        if bridged
                        else f"Edge routes cross: {first['id']} / {second['id']}"
                    ),
                    union_boxes(route_crossings),
                    edges=[first["id"], second["id"]],
                )

    for callout in measured["callouts"]:
        if not callout["frame"]:
            continue
        for node in nodes:
            if not node["frame"]:
                continue
            overlap = intersection(callout["frame"], node["frame"], tolerance=1.0)
            if overlap:
                add_issue(
                    issues,
                    "error",
                    "callout_node_collision",
                    f"Callout {callout['id']} overlaps node {node['id']}",
                    overlap,
                    callout=callout["id"],
                    node=node["id"],
                )
        for edge in edges:
            if not edge["labelRect"]:
                continue
            overlap = intersection(callout["frame"], edge["labelRect"], tolerance=1.0)
            if overlap:
                add_issue(
                    issues,
                    "error",
                    "callout_edge_label_collision",
                    f"Callout {callout['id']} overlaps edge label {edge['id']}",
                    overlap,
                    callout=callout["id"],
                    edge=edge["id"],
                )
    for header in measured["headers"]:
        if not header["bbox"]:
            continue
        for group in measured["groups"]:
            if not group["frame"]:
                continue
            overlap = intersection(header["bbox"], inflate(group["frame"], 1.0), tolerance=0.5)
            if overlap:
                add_issue(
                    issues,
                    "error",
                    "header_group_collision",
                    f"Header text collides with group {group['id']}: {header['text']}",
                    overlap,
                    group=group["id"],
                    text=header["text"],
                )

    view_box = measured["viewBox"]
    content = measured["content"]
    architecture = measured["architecture"] or content
    if not contains(view_box, content, tolerance=1.0):
        add_issue(
            issues,
            "error",
            "content_outside_canvas",
            "Rendered content extends outside the SVG viewBox",
            content,
            view_box=view_box,
        )
    margins = {
        "left": architecture["x"] - view_box["x"],
        "top": architecture["y"] - view_box["y"],
        "right": view_box["right"] - architecture["right"],
        "bottom": view_box["bottom"] - architecture["bottom"],
    }
    margin_ratios = {
        "left": margins["left"] / view_box["width"],
        "right": margins["right"] / view_box["width"],
        "top": margins["top"] / view_box["height"],
        "bottom": margins["bottom"] / view_box["height"],
    }
    occupancy = (architecture["width"] * architecture["height"]) / (view_box["width"] * view_box["height"])
    full_content_occupancy = (content["width"] * content["height"]) / (view_box["width"] * view_box["height"])
    oversized = {name: ratio for name, ratio in margin_ratios.items() if ratio > max_margin_ratio}
    if oversized or occupancy < min_occupancy:
        add_issue(
            issues,
            "warning",
            "excess_canvas_whitespace",
            f"Content occupancy is {occupancy:.1%}; oversized margins: {', '.join(oversized) or 'none'}",
            architecture,
            occupancy=round(occupancy, 4),
            margins={name: round(value, 3) for name, value in margins.items()},
            margin_ratios={name: round(value, 4) for name, value in margin_ratios.items()},
        )

    errors = sum(issue["severity"] == "error" for issue in issues)
    warnings = sum(issue["severity"] == "warning" for issue in issues)
    return {
        "status": "fail" if errors else "pass",
        "summary": {"errors": errors, "warnings": warnings, "issues": len(issues)},
        "metrics": {
            "view_box": view_box,
            "content_bbox": content,
            "architecture_bbox": architecture,
            "content_occupancy": round(full_content_occupancy, 4),
            "architecture_occupancy": round(occupancy, 4),
            "margins": {name: round(value, 3) for name, value in margins.items()},
            "margin_ratios": {name: round(value, 4) for name, value in margin_ratios.items()},
            "nodes": len(nodes),
            "edges": len(edges),
            "callouts": len(measured["callouts"]),
            "group_gaps": group_gaps,
        },
        "issues": issues,
    }


def write_overlay(svg_path: Path, overlay_path: Path, issues: list[dict[str, Any]]) -> None:
    tree = ET.parse(svg_path)
    root = tree.getroot()
    namespace = "http://www.w3.org/2000/svg"
    ET.register_namespace("", namespace)
    layer = ET.Element(f"{{{namespace}}}g", {"id": "geometry-check-overlay", "pointer-events": "none"})
    for issue in issues:
        bbox = issue.get("bbox")
        if not bbox:
            continue
        severity = issue["severity"]
        color = "#dc2626" if severity == "error" else "#d97706"
        width = max(float(bbox["width"]), 4.0)
        height = max(float(bbox["height"]), 4.0)
        x = float(bbox["x"]) - max(0.0, (4.0 - float(bbox["width"])) / 2)
        y = float(bbox["y"]) - max(0.0, (4.0 - float(bbox["height"])) / 2)
        ET.SubElement(
            layer,
            f"{{{namespace}}}rect",
            {
                "x": f"{x:.2f}",
                "y": f"{y:.2f}",
                "width": f"{width:.2f}",
                "height": f"{height:.2f}",
                "fill": color,
                "fill-opacity": "0.10",
                "stroke": color,
                "stroke-width": "3",
                "stroke-dasharray": "8 5" if severity == "warning" else "none",
                "data-issue": str(issue["kind"]),
            },
        )
    root.append(layer)
    overlay_path.parent.mkdir(parents=True, exist_ok=True)
    tree.write(overlay_path, encoding="utf-8", xml_declaration=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("svg", type=Path)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--overlay-out", type=Path)
    parser.add_argument("--screenshot-out", type=Path)
    parser.add_argument("--max-margin-ratio", type=float, default=0.22)
    parser.add_argument("--min-content-occupancy", type=float, default=0.50)
    parser.add_argument("--min-group-gap", type=float, default=40.0)
    parser.add_argument("--fail-on-error", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source_path = str(args.svg)
    svg_path = args.svg.resolve()
    if not svg_path.exists():
        print(f"missing SVG: {svg_path}", file=sys.stderr)
        return 2

    measured, browser_label = measure_svg(svg_path, args.screenshot_out)
    result = analyze(measured, args.max_margin_ratio, args.min_content_occupancy, args.min_group_gap)
    result["source"] = source_path
    result["browser"] = browser_label

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if args.overlay_out:
        write_overlay(svg_path, args.overlay_out, result["issues"])

    print(json.dumps({"status": result["status"], **result["summary"], **result["metrics"]}, indent=2))
    for issue in result["issues"]:
        print(f"[{issue['severity']}] {issue['kind']}: {issue['message']}")
    return 1 if args.fail_on_error and result["summary"]["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())

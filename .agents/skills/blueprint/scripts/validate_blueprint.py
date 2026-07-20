#!/usr/bin/env python3
"""Validate blueprint schema and optional adapter artifacts."""

from __future__ import annotations

import argparse
import json
import pathlib
from typing import Any

try:
    import yaml
except ImportError as exc:  # pragma: no cover - runtime guidance
    raise SystemExit(
        "Missing dependency: pyyaml. Run with: uv run --with pyyaml python validate_blueprint.py"
    ) from exc


def load_schema(path: pathlib.Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def fail(errors: list[str], message: str) -> None:
    errors.append(message)


def warn(warnings: list[str], message: str) -> None:
    warnings.append(message)


def validate_schema(schema: dict[str, Any], warnings: list[str], errors: list[str]) -> None:
    nodes = schema.get("nodes") or []
    edges = schema.get("edges") or []
    groups = schema.get("groups") or []
    callouts = schema.get("callouts") or []
    composition = schema.get("composition") or {}
    node_ids = [str(node.get("id", "")) for node in nodes]
    if len(node_ids) != len(set(node_ids)):
        fail(errors, "duplicate node ids")
    known = set(node_ids)
    if len(nodes) > 14:
        warn(warnings, f"large top-level node set may be a mega-diagram: {len(nodes)} nodes")
    if len(nodes) < 4:
        warn(warnings, f"top-level node set may be too small to explain a system: {len(nodes)} nodes")
    for node in nodes:
        insets = node.get("insets") or []
        if len(insets) > 6:
            warn(warnings, f"node has many insets; consider a drill-down diagram: {node.get('id')}")
    for edge in edges:
        source = str(edge.get("from", ""))
        target = str(edge.get("to", ""))
        if source not in known:
            fail(errors, f"edge source missing: {source}")
        if target not in known:
            fail(errors, f"edge target missing: {target}")
        label = str(edge.get("label") or edge.get("kind") or "").strip().lower()
        if label in {"use", "uses", "depends", "depends on"}:
            warn(warnings, f"generic relationship label: {source}->{target} '{label}'")
        if len(str(edge.get("label") or "")) > 36:
            warn(warnings, f"long edge label may clutter the diagram: {source}->{target}")
    for group in groups:
        ids = [str(x) for x in group.get("nodes", [])]
        for node_id in ids:
            if node_id not in known:
                fail(errors, f"group node missing: {group.get('id')}->{node_id}")
        layers = {
            str(node.get("layer") or "middle")
            for node in nodes
            if str(node.get("id", "")) in set(ids)
        }
        if len(layers) > 2:
            warn(warnings, f"group spans more than two layers and may overlap visually: {group.get('id')}")
    for callout in callouts:
        callout_id = callout.get("id", callout.get("label", "callout"))
        target = callout.get("target")
        if target and str(target) not in known:
            fail(errors, f"callout target missing: {callout_id}->{target}")
        for node_id in [str(x) for x in callout.get("nodes", [])]:
            if node_id not in known:
                fail(errors, f"callout node missing: {callout_id}->{node_id}")

    edge_pairs = {
        (str(edge.get("from", "")), str(edge.get("to", "")))
        for edge in edges
    }
    path_ids: list[str] = []
    allowed_roles = {"primary", "context", "secondary", "feedback"}
    for path in composition.get("paths") or []:
        path_id = str(path.get("id") or "")
        path_ids.append(path_id)
        role = str(path.get("role") or "")
        ids = [str(node_id) for node_id in path.get("nodes") or []]
        if not path_id:
            fail(errors, "composition path missing id")
        if role not in allowed_roles:
            fail(errors, f"composition path has invalid role: {path_id}->{role}")
        if len(ids) < 2:
            fail(errors, f"composition path needs at least two nodes: {path_id}")
        for node_id in ids:
            if node_id not in known:
                fail(errors, f"composition path node missing: {path_id}->{node_id}")
        for source, target in zip(ids, ids[1:]):
            if source in known and target in known and (source, target) not in edge_pairs:
                fail(errors, f"composition path edge missing: {path_id}->{source}->{target}")
    if len(path_ids) != len(set(path_ids)):
        fail(errors, "duplicate composition path ids")

    lane_ids: list[str] = []
    lane_membership: dict[str, str] = {}
    allowed_positions = {"left", "center", "right"}
    for lane in composition.get("lanes") or []:
        lane_id = str(lane.get("id") or "")
        lane_ids.append(lane_id)
        position = str(lane.get("position") or "")
        ids = [str(node_id) for node_id in lane.get("nodes") or []]
        if not lane_id:
            fail(errors, "composition lane missing id")
        if position not in allowed_positions:
            fail(errors, f"composition lane has invalid position: {lane_id}->{position}")
        if not ids:
            fail(errors, f"composition lane needs at least one node: {lane_id}")
        for node_id in ids:
            if node_id not in known:
                fail(errors, f"composition lane node missing: {lane_id}->{node_id}")
            previous_lane = lane_membership.get(node_id)
            if previous_lane and previous_lane != lane_id:
                fail(errors, f"composition lane node repeated: {node_id}->{previous_lane},{lane_id}")
            lane_membership[node_id] = lane_id
        lane_layers = [
            str(node.get("layer") or "middle")
            for node in nodes
            if str(node.get("id") or "") in set(ids)
        ]
        if len(lane_layers) != len(set(lane_layers)):
            warn(warnings, f"composition lane has multiple nodes in one layer: {lane_id}")
    if len(lane_ids) != len(set(lane_ids)):
        fail(errors, "duplicate composition lane ids")


def validate_excalidraw(path: pathlib.Path, warnings: list[str], errors: list[str]) -> None:
    if not path.exists():
        return
    try:
        scene = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        fail(errors, f"invalid Excalidraw JSON: {exc}")
        return
    if scene.get("type") != "excalidraw":
        fail(errors, "Excalidraw scene type is not 'excalidraw'")
    elements = scene.get("elements") or []
    ids = [element.get("id") for element in elements]
    if len(ids) != len(set(ids)):
        fail(errors, "duplicate Excalidraw element ids")
    if len(elements) > 180:
        warn(warnings, f"large Excalidraw scene may be hard to edit: {len(elements)} elements")
    unreadable = [
        element.get("id")
        for element in elements
        if element.get("type") == "text" and int(element.get("fontSize") or 0) < 12
    ]
    if unreadable:
        warn(warnings, f"small Excalidraw text elements: {len(unreadable)}")


def validate_files(base: pathlib.Path, warnings: list[str], errors: list[str]) -> None:
    for rel in ["diagram.svg", "diagram.html", "diagram.png"]:
        path = base / rel
        if not path.exists() or path.stat().st_size == 0:
            fail(errors, f"missing or empty {rel}")
    for rel in ["diagram.png", "diagram.render.png", "diagram.generated.png"]:
        png = base / rel
        if png.exists() and png.stat().st_size == 0:
            fail(errors, f"empty {rel}")
        elif png.exists() and png.stat().st_size < 10_000:
            warn(warnings, f"{rel} looks too small: {png.stat().st_size} bytes")
    validate_excalidraw(base / "diagram.excalidraw", warnings, errors)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--schema", default="schema.yaml")
    parser.add_argument("--artifacts-dir", default=".")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    warnings: list[str] = []
    errors: list[str] = []
    schema_path = pathlib.Path(args.schema)
    validate_schema(load_schema(schema_path), warnings, errors)
    validate_files(pathlib.Path(args.artifacts_dir), warnings, errors)
    result = {"valid": not errors, "errors": errors, "warnings": warnings}
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(f"valid={result['valid']}")
        for item in errors:
            print(f"ERROR: {item}")
        for item in warnings:
            print(f"WARN: {item}")
    raise SystemExit(0 if result["valid"] else 1)


if __name__ == "__main__":
    main()

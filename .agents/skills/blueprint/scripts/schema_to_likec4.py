#!/usr/bin/env python3
"""Export a blueprint schema to a starter LikeC4/C4 model.

The output is a model-discipline adapter: it gives stable IDs, domain labels,
descriptions, technologies, and directional relationships that can be refined
with LikeC4 tooling. The blueprint schema remains authoritative unless the
project explicitly adopts LikeC4 as its source of truth.
"""

from __future__ import annotations

import argparse
import pathlib
import re
from typing import Any

try:
    import yaml
except ImportError as exc:  # pragma: no cover - runtime guidance
    raise SystemExit(
        "Missing dependency: pyyaml. Run with: uv run --with pyyaml python schema_to_likec4.py"
    ) from exc


def load_schema(path: pathlib.Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def q(value: Any) -> str:
    return str(value or "").replace("\\", "\\\\").replace('"', '\\"')


def ident(value: Any) -> str:
    text = re.sub(r"[^A-Za-z0-9_]+", "_", str(value or "").strip())
    text = re.sub(r"_+", "_", text).strip("_").lower()
    if not text:
        text = "item"
    if text[0].isdigit():
        text = f"n_{text}"
    return text


def rel_label(edge: dict[str, Any]) -> str:
    label = str(edge.get("label") or edge.get("kind") or "relates to")
    bad = {"uses", "use", "depends", "depends on"}
    return "relates to" if label.lower().strip() in bad else label


def export_model(schema: dict[str, Any]) -> str:
    meta = schema.get("meta") or {}
    title = q(meta.get("title") or "Architecture")
    nodes = schema.get("nodes", [])
    edges = schema.get("edges", [])
    node_ids = {str(node["id"]): ident(node["id"]) for node in nodes}

    lines: list[str] = [
        "specification {",
        "  element actor",
        "  element system",
        "  element container",
        "  element component",
        "}",
        "",
        "model {",
        f'  product = system "{title}" {{',
        f'    description "{q(meta.get("subtitle") or "Generated from blueprint schema.yaml")}"',
        "",
    ]

    for node in nodes:
        node_name = node_ids[str(node["id"])]
        label = q(node.get("label") or node["id"])
        signature = q(node.get("signature") or node.get("category") or "")
        note = q(node.get("note") or "")
        lines.append(f'    {node_name} = component "{label}" {{')
        if signature:
            lines.append(f'      technology "{signature}"')
        if note:
            lines.append(f'      description "{note}"')
        lines.append("    }")
        lines.append("")

    lines.append("  }")
    lines.append("")

    for edge in edges:
        source = node_ids.get(str(edge.get("from")))
        target = node_ids.get(str(edge.get("to")))
        if not source or not target:
            continue
        lines.append(f'  product.{source} -> product.{target} "{q(rel_label(edge))}"')

    lines.extend(
        [
            "}",
            "",
            "views {",
            "  view context {",
            '    title "System Context"',
            "    include product",
            "  }",
            "",
            "  view containers {",
            '    title "Container / Component View"',
            "    include product.*",
            "    autoLayout TopBottom",
            "  }",
            "}",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--schema", default="schema.yaml")
    parser.add_argument("--out", default="model.c4")
    args = parser.parse_args()

    model = export_model(load_schema(pathlib.Path(args.schema)))
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(model, encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

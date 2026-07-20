# Blueprint Adapter Matrix

Use this reference when choosing how a blueprint should produce visual artifacts.

## Roles

| Adapter | Best For | Source Of Truth | Editable | Maturity Signal | Use With Blueprint |
|---|---|---|---|---|---|
| Native render | Durable project architecture maps | `schema.yaml` | Yes, via schema/layout/renderer | Geometry and schema validation | Authoritative maintenance track |
| Excalidraw | Human-editable whiteboard diagrams | `.excalidraw` JSON, derived from schema | Yes, in Excalidraw | Mature external skill patterns | Optional editable collaboration artifact |
| LikeC4 / C4 DSL | Long-lived architecture-as-code views | `.c4` / `.likec4` model | Yes, as DSL | Validation-backed external skill | Optional model discipline for complex systems |
| Standalone SVG / HTML | Polished presentation diagrams | SVG/HTML source | Partly, by editing markup | Mature external SVG/HTML diagram skills | Presentation artifact, not sole truth |
| Generated image | Presentation composition and visual hierarchy | Prompt + source schema/composition | No | Side-by-side semantic review | Parallel presentation candidate when available |
| Mermaid | Lightweight fallback preview | `diagram.mmd`, derived from schema | Yes, text edit | Ubiquitous but visually limited | Fallback view only |

## Borrowed Practices

- From Excalidraw diagram skills: keep valid `.excalidraw` JSON when the user needs manual editing; keep element counts bounded; validate IDs, text readability, and arrow logic.
- From LikeC4 architecture skills: model structure before visuals; use stable IDs, domain names, directional labeled relationships, focused views, and CLI validation when a DSL is produced.
- From SVG/HTML architecture skills: use semantic colors, region boundaries, strict spacing, arrows behind nodes, legends outside boundaries, and render/view/fix loops.

## Decision Rules

- For human-facing architecture work, run native render and generated-image
  tracks in parallel when image generation is available. Do not wait for the
  render to fail before trying the generated composition.
- Preserve separate candidates as `diagram.render.png` and
  `diagram.generated.png`; use `diagram.png` only for the reviewed presentation
  selection.
- If the user says "editable final architecture diagram", produce native render first and Excalidraw as the editable handoff when useful.
- If the project is large or will be maintained by multiple agents, add LikeC4/C4 model output or at least C4-style view names.
- If the user only needs a nice report visual, SVG/HTML or generated PNG is acceptable, but label it as presentation.
- If only Mermaid exists, mark the diagram as a fallback and list missing final artifacts.

## Quality Gate

A blueprint diagram is not finished until the deliverables state:

1. The authoritative source.
2. Which adapters were produced.
3. Which adapters were intentionally skipped and why.
4. Whether the rendered visual was inspected.
5. Whether both default tracks ran, or why one was skipped.
6. Which candidate was selected as `diagram.png` and why.
7. Whether the diagram is final, collaboration-ready, presentation-only, or fallback-only.

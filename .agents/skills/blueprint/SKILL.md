---
name: blueprint
description: "Three-view architecture documentation pattern for project blueprints. Use when creating or updating architecture docs, diagrams, schema-backed visual maps, BLUEPRINT.md files, module maps, or architecture review ledgers. Keeps deterministic maintenance renders and generated presentation diagrams as parallel tracks when image generation is available."
---

# Blueprint -- Architecture Map Workflow

## Core Position

Use this skill to create and maintain project architecture blueprints with an explicit source of truth, a readable fallback diagram, and a reviewed visual output.

For human-facing architecture work, the preferred mature form is a parallel
two-track workflow:

```text
schema.yaml -> deterministic renderer -> diagram.svg / diagram.html / diagram.render.png
            -> image generation       -> diagram.generated.png
            -> visual review          -> diagram.png (selected presentation view)
            -> diagram.mmd fallback + BLUEPRINT.md narrative
```

When image generation is available and the user has not opted out, run the
render and generate tracks from the same reviewed schema and composition plan.
Do not make the generated track wait for the render to fail. The two tracks
solve different problems:

- **Render track** = authoritative maintenance version. Deterministic, reviewable, reproducible, and suitable for long-lived project truth.
- **Excalidraw track** = editable collaboration version. Useful when humans need to move labels/boxes directly, but it should be derived from or reconciled with the schema.
- **LikeC4/C4 track** = architecture-as-code version. Useful for larger systems where context/container/component views and validation matter.
- **SVG/HTML presentation track** = polished visual version. Useful for reports, but not enough as the sole source of truth.
- **Generate track** = parallel presentation candidate. It often produces stronger global spacing, hierarchy, and composition, but it is not an editable source of architecture truth.
- **Mermaid track** = fallback view. Good for GitHub/Obsidian preview and quick editing, but usually not good enough as the main architecture image.

`diagram.png` is the selected human-facing view, not a source of truth. When
both tracks run, preserve `diagram.render.png` and `diagram.generated.png`
separately, then choose or copy the reviewed winner to `diagram.png`.

## When To Use

Use this skill when the user asks for any of:

- creating or updating `docs/architecture/`
- creating a project blueprint or `BLUEPRINT.md`
- adding/removing/renaming architecture modules
- making an editable architecture diagram
- producing a schema-backed architecture map
- reviewing whether an architecture diagram is good enough
- comparing against approved examples recorded in `references/example_registry.md`

## Required Output Files

For a real project blueprint, create or maintain:

```text
docs/architecture/
├── schema.yaml              # Source of Truth
├── diagram.mmd              # Mermaid fallback view
├── diagram.svg              # editable deterministic render, preferred
├── diagram.html             # optional interactive/render preview
├── diagram.render.png       # deterministic raster candidate
├── diagram.generated.png    # generated presentation candidate, when available
├── diagram.png              # reviewed human-facing selection
├── diagram.excalidraw       # optional human-editable collaboration export
├── model.c4 / model.likec4  # optional architecture-as-code model
└── render.py                # optional project-local renderer wrapper

BLUEPRINT.md                 # architecture image + module/data-contract narrative
```

For a lightweight first pass, `schema.yaml + diagram.mmd + BLUEPRINT.md` is acceptable, but mark the missing renderer/PNG as a known gap. Do not pretend a Mermaid screenshot is the final architecture map if the user cares about editable diagrams.

## Decision Rules

Before choosing an output format, read `references/adapter_matrix.md` when:

- the user compares "render" with "generated"
- the user asks whether to borrow another diagram skill
- the user asks for an editable final diagram
- multiple versions of this skill are being evaluated

Before choosing top-level architecture nodes, read `references/abstraction_lens.md`.
This is mandatory when the task compares prompt families, references GT diagrams,
or asks whether a blueprint shows real architecture blocks rather than folders.
It is also mandatory for early projects with existing layer/module docs: recover
candidate blocks and edges from the repo first, then ask only about unresolved
maintainer decisions. Do not turn numbered layers into a serial pipeline without
checking whether the topology is hub-and-spoke, read-only context, feedback loop,
or taxonomy.

### Always Keep The Render Track

Keep deterministic rendering when:

- the diagram is expected to live in the repo
- the user says "editable", "real architecture diagram", "可编辑", or compares to arxiv-marker
- the diagram will be revised over time
- multiple agents or humans need to review architecture drift
- schema/layout consistency matters more than instant polish

Use `scripts/render_architecture.py` as a reusable starting renderer. Copy or reference it into the project when the blueprint needs a deterministic visual:

```bash
uv run --with pyyaml --with cairosvg python docs/architecture/render.py \
  --schema docs/architecture/schema.yaml \
  --out-dir docs/architecture \
  --png \
  --png-name diagram.render.png
```

If no project-local renderer exists yet, copy the skill script:

```text
<skill>/scripts/render_architecture.py -> docs/architecture/render.py
```

Then adjust only project-specific layout and styling in the project copy.

If PNG rendering through CairoSVG fails, especially on Windows where native Cairo may be absent, keep the SVG/HTML as the deterministic render and create PNG from the HTML preview with the browser fallback:

```bash
python <skill>/scripts/html_to_png.py docs/architecture/diagram.html docs/architecture/diagram.render.png --width 1800 --height 1400
```

Record which path produced the PNG. Do not silently claim CairoSVG succeeded when a browser screenshot fallback was used.

### Add Excalidraw Track For Human Editing

Add `diagram.excalidraw` when:

- the user explicitly wants a diagram that can be opened and edited by hand
- the diagram is a collaboration artifact, not only a repo truth artifact
- the final visual needs labels/placement tuned interactively

Keep Excalidraw as an adapter, not the only truth, unless the user explicitly chooses it as the source. Validate that JSON is parseable, text is readable, element IDs are unique, and arrows land on intended nodes.

Use `scripts/schema_to_excalidraw.py` for a deterministic first editable handoff:

```bash
uv run --with pyyaml python <skill>/scripts/schema_to_excalidraw.py --schema docs/architecture/schema.yaml --out docs/architecture/diagram.excalidraw
```

After export, open or inspect a rendered preview if available. Treat manual Excalidraw edits as downstream changes that should be reconciled back to `schema.yaml` when they change architecture facts.

### Add LikeC4/C4 Track For Model Discipline

Add `model.c4` or `model.likec4` when:

- the system is large enough to benefit from separate context/container/component views
- architecture drift should be validated by CLI
- the user wants architecture-as-code rather than only a graphic

Use stable IDs, domain-level names, explicit directional relationships, and focused views. Prefer relationship labels such as `reads`, `writes`, `uploads`, `renders`, `filters`, `geocodes`, and `authenticates` over generic `uses`.

Use `scripts/schema_to_likec4.py` for a starter model:

```bash
uv run --with pyyaml python <skill>/scripts/schema_to_likec4.py --schema docs/architecture/schema.yaml --out docs/architecture/model.c4
npx likec4 validate docs/architecture --json
```

Do not validate the file path alone. `npx likec4 validate docs/architecture/model.c4` can exit 0 while reporting zero source files. Treat LikeC4 as validated only when JSON output has `valid: true`, `stats.totalFiles > 0`, and `stats.totalErrors == 0`.

If validation fails because the project lacks LikeC4 setup or the DSL needs refinement, report the exact failure and keep `model.c4` marked as a starter/candidate artifact, not a validated final model.

### Add SVG/HTML Presentation Track For Report-Ready Output

Use standalone SVG/HTML when:

- the user needs a polished report artifact
- the artifact should open directly in a browser
- export buttons or presentation cards are useful

Do not treat an HTML/SVG presentation as authoritative unless it is generated from the schema or has a clearly documented source.

### Visual Acceptance Gate

Keep semantic validation separate from visual acceptance. A diagram can pass
schema validation, LikeC4 validation, file-existence checks, and screenshot
nonblank checks while still failing as a presentation artifact.

Before calling a rendered reconstruction "presentation-ready":

- Run `scripts/check_svg_geometry.py` when Playwright/Chromium is available.
- Inspect its overlay for text overflow, node or group collisions, weak group
  spacing, edge crossings, route/label collisions, and clipped canvas content.
- Treat unintended diagonal segments as errors; deterministic architecture
  routes should use a consistent orthogonal visual language.
- Treat a plain interior edge crossing as an error. If a complex maintenance
  view cannot avoid one without a much worse detour, render an explicit line
  bridge and keep the resulting `bridged_edge_crossing` warning visible.
- Compare it side by side with the best approved generated/reference image.
- Ask whether a human would prefer the render for a report or slide without
  extra explanation.
- Check visual hierarchy, spacing, rhythm, typography, edge routing, and whether
  the image feels composed rather than merely non-overlapping.
- If the generated reference is clearly better, keep it as the presentation
  artifact and mark the render as a semantic/maintenance draft.
- Do not let editability language hide a weaker visual result. Say "validated
  but not presentation-ready" when that is the honest verdict.

For Gen-to-Render work, the target is not only "same nodes and edges"; it is
visual parity with the chosen generated concept. If parity is not reached, the
workflow is still useful, but the output should not be promoted as the main
human-facing diagram.

Example geometry check:

```bash
python <skill>/scripts/check_svg_geometry.py \
  docs/architecture/diagram.svg \
  --json-out docs/architecture/geometry.json \
  --overlay-out docs/architecture/geometry-overlay.svg \
  --fail-on-error
```

Geometry-clean is necessary, not sufficient. A maintainer should still decide
whether the deterministic render or an approved generated/reference image is
the better presentation artifact.

### Add Change / Focus Views For Local Work

Use a change/focus view when the user is discussing a specific implementation change, review area, or ownership slice on top of an otherwise stable architecture.

Keep the canonical macro graph stable. A frontend change, for example, should usually highlight the read-side blocks or open a frontend subview; it should not redraw write-side or data-plane architecture unless the change alters a cross-block contract.

Recommended pattern:

```text
canonical schema -> neutral macro diagram
                 -> focus overlay variant
                 -> optional subview/drill-down for the affected macro block
```

Use focus overlays for temporary review context:

- Highlight affected macro nodes or regions.
- Dim unaffected regions when that improves scanning.
- Add A/B/C markers for touched blocks.
- Open a local subview inside the affected macro block when the user needs implementation detail.
- Keep edge contracts in the report ledger when labels would clutter the image.

Do not encode a focus overlay as a permanent architecture fact unless it changes responsibilities, data contracts, access boundaries, durable state, deployment boundaries, or external dependencies.

### Run Generate Track In Parallel For Presentation

Run a generated presentation candidate alongside the render when image
generation is available and any of these are true:

- the user wants a quick "looks good" visual
- the diagram is for a report or slide
- the architecture is still exploratory
- the user values spatial composition or explicitly prefers a generated look

Use the same `schema.yaml`, edge ledger, `composition.paths`, and
`composition.lanes` for both tracks. Feed approved style references only as
visual guidance; never use them to invent nodes or edges. Save the result as
`diagram.generated.png`, validate its node/label/edge fidelity, and compare it
side by side with `diagram.render.png`.

The generated candidate may become the main report or slide image when it is
visually stronger. Still keep `schema.yaml`, `diagram.svg`, and the deterministic
render as the maintenance artifacts. If image generation is unavailable or
fails after a reasonable attempt, finish the render track and record the
generated track as skipped or failed rather than hiding it.

### Use Mermaid Only As Fallback

Use Mermaid for:

- GitHub/Obsidian preview
- quick review in `diagram.mmd`
- diff-friendly fallback

Avoid using Mermaid-exported PNG as the main deliverable when the user asks for a polished or editable architecture map. A Mermaid screenshot is usually an anti-pattern for final blueprint visuals.

## Schema Rules

Use `templates/schema.yaml` as the base. Keep these semantics:

- `meta`: title, subtitle, updated date, direction, optional version
- `categories`: semantic visual classes, not color names
- `nodes`: stable architecture components with `id`, `label`, `category`, optional `signature`, `note`, `shape`, `style`, `layer`, `align`, `expand`
- `edges`: relationships with `from`, `to`, `kind`, optional `label`, `style`
- `composition`: optional semantic-to-spatial plan. `paths` classify connected node sequences as `primary`, `context`, `secondary`, or `feedback`; `lanes` keep related nodes in relative `left`, `center`, or `right` positions without storing pixel coordinates

Optional renderer-oriented fields are allowed when needed:

- `groups`: visual regions such as "frontend entry", "core pipeline", "external write boundary"
- `callouts`: small chips for runtime constraints, manual fallbacks, side services, or caveats that should not become main nodes
- `insets`: small subcards inside a macro node, for 2-5 local substeps that clarify the block without promoting them to top-level architecture
- `layout`: per-node explicit `x`, `y`, `w`, `h` overrides when automatic layout is not enough
- `icon`: renderer hint, such as `terminal`, `database`, `cloud`, `file`, `chart`

Keep `diagram.mmd` as a subset of schema. Keep `diagram.svg/png/html` as rendered views, not independent truth.

Prefer `composition` before explicit `layout`. Composition preserves the
architectural reading order across deterministic render and generated-image
tracks, while coordinates remain a renderer concern. Use `layout` only after
the shared plan is correct and a project still needs local geometry tuning.

Choose nodes with the abstraction lens:

- Start macro-first: system boundary, 4-7 maintainer-facing work divisions or data planes, then the edges between them.
- For early projects, read existing plan/architecture docs first and preserve maintainer vocabulary as candidate intent.
- Present candidate blocks, candidate edges, evidence, and open decisions before drawing when the macro boundary is not yet reviewed.
- Treat layer names as topology-neutral until evidence shows whether they are pipeline steps, hubs, read-only context planes, feedback loops, or taxonomy.
- Use main nodes for domain, workflow, data, contract, external dependency, review, storage, and mutation boundaries.
- Work-division blocks are valid when they have handoff meaning: ownership, input/output contract, review criteria, state mutation, deployment, or access consequences.
- Put file paths, functions, tables, and implementation details inside `signature` or `note` unless the file itself is the user-facing architecture boundary.
- If a macro block looks too empty, use `insets` or `expand` drill-downs before splitting it into same-weight top-level nodes.
- Keep the top-level view compact. Around 8-12 primary cards is usually enough; if more than roughly 14 are required, split the view or add `expand` drill-downs.
- Planned or not-yet-built boundaries may be shown only when repo docs/product contract make them part of the architecture question. Mark them with `style: dashed` and explain the status in `note`.
- For code-change discussion, prefer a focus overlay or affected-block subview over changing the canonical macro graph.

Keep group boundaries visually meaningful. Avoid groups that span many unrelated layers because they create large overlapping frames. If a group would span more than two layers, prefer a narrative section in `BLUEPRINT.md` or split the group.

For Chinese/CJK labels, widen boxes or shorten signatures/notes. A final diagram must not have text clipped by card edges, hidden below the viewport, or labels that overlap following content.

## BLUEPRINT.md Requirements

Do not stop at the image. A useful blueprint normally includes:

1. System architecture image and source links.
2. One-paragraph positioning.
3. Module definitions: responsibility, interface, input, output, non-goals.
4. Data contracts or schemas at module boundaries.
5. Key design choices and boundaries.
6. Current gaps / review status.

Use `templates/blueprint_section.md` only for the architecture image section. Extend it with module and contract narrative when the project needs a real blueprint.

## Example Library And Review Status

Use `references/example_registry.md` before claiming a diagram style is "approved".

The bundled registry contains pattern-level examples, not required local files.
Use it to recognize useful abstraction shapes such as editable-renderer,
presentation-style, module-contract, macro-data-plane, and anti-pattern examples.

For a project-specific installation, add examples only after review. Do not bind
the skill to private absolute paths, unreviewed generated images, or one
maintainer's local workspace.

When adding a new example, record its status:

```yaml
project: <name>
path: <file-or-folder>
status: approved | candidate | rejected
good_for: editable-renderer | presentation-style | module-contract | anti-pattern
reason: <short verdict>
```

## Workflows

### New Project Blueprint

1. Inspect the project structure and existing docs.
2. If the repo already has layer/module/roadmap docs, extract candidate maintainer vocabulary before inventing new block names.
3. Identify the macro architecture boundary using `references/abstraction_lens.md`: system purpose, outside inputs, durable data plane, write/mutation side, read/experience side, access/privacy boundary, external services, and planned boundaries only when they are real product contracts.
4. Classify the topology before drawing arrows: pipeline, hub-and-spoke, read-only context plane, feedback loop, taxonomy, or a mix.
5. Draft a candidate review ledger before drawing: `candidate block`, `role`, `evidence`, `proposed edges`, and `open decision`.
6. Draft a macro edge ledger before drawing: `from`, `to`, `contract/artifact`, `evidence`, and whether it is main path, side service, read-only context, or feedback.
7. Draft a composition plan before coordinates: identify primary, context,
   secondary, and feedback paths, then assign only the relative lanes needed to
   keep the main reading order clear. When ownership, topology, a contract, or
   the macro arrangement is ambiguous, show the candidate blocks, edge ledger,
   and composition options to the maintainer and pause before drawing. Skip the
   pause only when the evidence is unambiguous.
8. Choose 4-7 top-level work divisions or data planes. Prefer "here is the
   candidate arrangement; does this match your intent?" over broad open-ended
   questions.
9. Use `insets`, `callouts`, or `expand` drill-downs for local detail instead of promoting every helper into a top-level node.
10. Create `docs/architecture/schema.yaml`, including `composition` when the topology needs a stable reading order.
11. Create `docs/architecture/diagram.mmd` as fallback.
12. Prefer copying `scripts/render_architecture.py` to `docs/architecture/render.py`; render `diagram.svg`, `diagram.html`, and `diagram.render.png`.
13. When image generation is available and the output is human-facing, run `templates/image_prompt.md` from the same schema/composition plan and save `diagram.generated.png`.
14. Decide whether to add adapter artifacts: `diagram.excalidraw`, `model.c4`, or another standalone view.
15. Generate chosen adapters with scripts when available instead of hand-writing large JSON/DSL from scratch.
16. Create or update `BLUEPRINT.md` with the image, source links, modules, data contracts, adapter status, and known gaps.
17. Run `scripts/validate_blueprint.py` against the schema and artifact directory when available.
18. Validate that rendered output is not blank, node/edge counts match schema, and chosen adapters are parseable/validated.
19. Run the browser geometry checker when available; inspect both the machine
    result and overlay, then refine layout until errors are cleared or recorded.
20. Compare `diagram.render.png` and `diagram.generated.png` when both exist. Validate semantic fidelity, select the stronger human-facing view as `diagram.png`, and record why.
21. If the render is geometry-clean but compositionally weaker, keep it as the maintenance view and let the generated candidate carry presentation duties.

### Update Existing Architecture

1. Edit `schema.yaml` first.
2. Sync `diagram.mmd`.
3. Re-run the deterministic renderer if present.
4. Re-run the generated presentation track when image generation is available
   and the project has a human-facing architecture image.
5. Recompare both candidates and update `diagram.png` only after review.
6. Update `BLUEPRINT.md` narrative when module boundaries or contracts changed.
7. Report exactly which views were updated, selected, skipped, or remain stale.

### Review A Diagram

Judge with this checklist:

- Is there a clear source of truth?
- Can the main diagram be edited deterministically?
- Is there a human-editable handoff if the user asked for one?
- Are views split by architecture question rather than crammed into one giant map?
- Does the picture show architecture modules, not just files?
- Did the diagram start from macro blocks and macro edges, rather than local files/components?
- Are work-division blocks meaningful as handoffs, not just labels?
- Are folders/files used as evidence inside cards instead of fake top-level blocks?
- Are boundaries and data contracts visible?
- Are relationship labels specific verbs rather than generic "uses" arrows?
- Are planned/roadmap blocks visibly dashed or otherwise lower weight?
- Are side services/config/runtime knobs represented as callouts when they would clutter the main map?
- Are generated/demo diagrams labelled as such?
- If the view is for a specific change, are focus markers separated from canonical architecture facts?
- Is there a module narrative below the image?
- Are unreviewed examples kept out of approved defaults?
- Was the rendered visual inspected for overlap, clipping, bad arrow routes, and weak hierarchy?
- If this is a render reconstruction of a stronger generated image, did it reach
  presentation parity? If not, is it labelled as a maintenance draft rather
  than the main human-facing diagram?
- Did LikeC4 validation inspect at least one source file, instead of returning a false green on a file path?

## Anti-Patterns

Avoid:

- treating a generated PNG as the only architecture artifact
- treating a weaker deterministic render as presentation-ready merely because it is editable or validator-clean
- treating a Mermaid screenshot as a polished final architecture map
- ignoring existing maintainer layer/module docs and asking the user broad architecture questions that files already answer
- turning numbered layers into a simple `1 -> 2 -> 3 -> ...` chain before checking whether the topology is hub, context, taxonomy, or feedback
- maintaining `schema.yaml`, `diagram.mmd`, and `diagram.png` manually without saying which one is authoritative
- grouping the system by folders or component names when the real boundaries are workflow, data contracts, storage, mutation, model quality, or human review
- drawing every helper module as a same-weight node
- hiding planned status on roadmap nodes
- filling the diagram with side services that would be clearer as callouts
- adding unreviewed generated examples as "approved"
- making architecture docs that contain a picture but no module/data-contract narrative
- hiding the architecture entry only in a subfolder without `BLUEPRINT.md`

## Naming

- Use `docs/architecture/`, not `docs/blueprint/`.
- Use lowercase stable schema node ids.
- Use semantic category names such as `front`, `data`, `pipeline`, `boundary`, not color names.
- Commit architecture changes as one conceptual unit, for example `arch: add editable architecture blueprint`.

## Bundled Resources

- `templates/schema.yaml`: baseline schema template.
- `templates/diagram.mmd`: Mermaid fallback template.
- `templates/blueprint_section.md`: minimal `BLUEPRINT.md` image/source section.
- `templates/image_prompt.md`: generated presentation diagram prompt, for generate track only.
- `scripts/render_architecture.py`: deterministic SVG/HTML/PNG renderer starter.
- `scripts/html_to_png.py`: Chrome/Edge headless PNG fallback for rendered HTML/SVG.
- `scripts/validate_blueprint.py`: schema/artifact validation helper.
- `scripts/check_svg_geometry.py`: optional Playwright/Chromium SVG geometry gate.
- `scripts/schema_to_excalidraw.py`: schema-derived editable Excalidraw handoff exporter.
- `scripts/schema_to_likec4.py`: schema-derived LikeC4/C4 starter model exporter.
- `references/example_registry.md`: approved/candidate/rejected example ledger.
- `references/adapter_matrix.md`: routing guide for native render, Excalidraw, LikeC4/C4, SVG/HTML, generated image, and Mermaid outputs.
- `references/abstraction_lens.md`: rules for selecting true architecture blocks instead of folder/component groupings.

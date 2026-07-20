# Blueprint Example Registry

This registry prevents the skill from learning unreviewed diagram noise as a
default style.

The public version intentionally stores pattern-level examples instead of
private absolute paths. Project-local installs may add concrete file paths after
the maintainer has reviewed them.

## Approved Pattern Examples

### Paper Resolver / Bibliography Enrichment

```yaml
project: paper-resolver-fixture
path: examples/paper-resolver/docs/architecture/
status: pattern-approved
good_for: editable-renderer
reason: Uses schema.yaml plus a deterministic renderer to produce a maintainable card-style diagram.
```

Use this pattern when the user wants:

- a real editable architecture diagram
- deterministic regeneration
- schema-backed visual truth
- SVG/HTML/PNG generated from the same source

Architecture lesson:

- The useful boundary is not the package layout.
- Good top-level blocks are entry adapters, external metadata sources, resolver
  cascade, ranking/proposal logic, human review/report products, and guarded
  write-back into the external system.
- Review and idempotent write guards are architecture boundaries, not incidental
  helper details.

### Social Sentiment Monitor

```yaml
project: sentiment-monitor-fixture
path: examples/sentiment-monitor/docs/architecture/
status: pattern-approved
good_for: presentation-style, module-contract
reason: Demonstrates a product/data-contract chain with a strong presentation visual and explicit module contracts.
```

Use this pattern when the user wants:

- a report-ready architecture image
- a clean module/data-contract blueprint narrative
- a generated or polished presentation view that still has a semantic source

Architecture lesson:

- The useful abstraction is a product/data contract chain:
  `watchlist -> scraper -> scorer -> aggregator -> metric store -> dashboard/notify/backtest`.
- Contracts such as `RawItem`, `ScoredItem`, and `DailyMetric` matter more than
  source file names.
- Model-quality and calibration loops are real architecture boundaries when
  they affect downstream decisions.

### Travel / Map Data Product

```yaml
project: travel-map-fixture
path: examples/travel-map/docs/architecture/
status: pattern-approved
good_for: editable-renderer, macro-data-plane, presentation-style-reference
reason: Demonstrates a larger app split between write-side curation, local processing, canonical data plane, frontend read experience, access boundary, and verification workflow.
```

Use this pattern when the user wants:

- a larger repo example that still has a schema-backed editable final diagram
- a macro-first split between write side, durable data plane, read side, and
  access boundary
- a comparison between maintainable render output and stronger presentation
  imagery

Architecture lesson:

- Keep the canonical data plane visible.
- Separate local ingestion/curation from frontend read experience.
- Treat access/privacy scope and workflow verification as architecture concerns
  when they shape what data can be seen or changed.

## Candidate Pattern Examples

### Early Portfolio / Decision Workflow

```yaml
project: portfolio-decision-fixture
path: examples/portfolio-decision/docs/architecture/
status: candidate
good_for: early-project-topology, candidate-first-review, read-only-context-plane
reason: Useful stress case where existing docs define numbered layers but the truthful topology is not a simple serial chain.
```

Use this when testing whether the skill:

- reads existing base architecture docs before inventing module names
- proposes candidate blocks and candidate edges instead of asking broad blank
  questions
- distinguishes numbered layers from pipeline order
- labels market/provider context as read-only when it must not become canonical
  account truth
- records pending maintainer review when the diagram is agent-drafted

Important notes:

- This is not an approved final example until a maintainer reviews the actual
  fixture output.
- Generated reference images are useful for visual direction, not edge truth.

### Pending Generated Diagrams

```yaml
project: varies
path: varies
status: candidate
good_for: pending-review
reason: Generated diagrams can be useful, but should not become defaults until each one has a review verdict.
```

Use generated examples only after recording a specific review verdict:

```yaml
project: <name>
path: <exact path>
status: approved | candidate | rejected
good_for: editable-renderer | presentation-style | module-contract | anti-pattern
reason: <why this diagram should or should not guide future blueprints>
```

## Anti-Pattern Examples

### Mermaid Screenshot As Final Diagram

```yaml
status: rejected
good_for: anti-pattern
reason: Information may be correct, but the PNG is not the editable final architecture map and often has weak layout/readability.
```

Mermaid remains useful as fallback, but do not stop there when the user asks for
polished or editable architecture docs.

### Folder-Shaped Architecture

```yaml
status: rejected
good_for: anti-pattern
reason: Top-level boxes are directories, packages, or UI components rather than maintainable boundaries.
```

Use file and folder names as evidence inside cards unless the file or folder is
itself the boundary being reviewed.

## Review Rubric

A diagram can become approved only if it passes most of these:

- Has a clear source of truth.
- Has an editable or deterministic main diagram route.
- Shows architecture modules, not just file names.
- Makes data flow and control flow visible.
- Makes safety/boundary modules visible when relevant.
- Has module definitions and data contracts in `BLUEPRINT.md` or equivalent
  docs.
- Has a known regeneration command.
- Avoids stale generated images with no editable backing.

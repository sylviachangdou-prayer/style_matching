# Blueprint Abstraction Lens

Use this reference before choosing top-level nodes for a blueprint diagram or
before reviewing whether a candidate diagram has real architecture blocks.

The goal is to recover system, workflow, data, and contract boundaries from the
repository. Do not promote folders, files, classes, or UI components to main
architecture blocks unless they are the actual boundary being reviewed.

## Existing Architecture First

Before inventing new macro blocks, search the repo's own planning and architecture
docs for existing layer, module, domain, or workflow language. Treat those docs as
candidate maintainer intent, then verify them against code and artifacts.

When existing docs define layers or macro blocks:

- Preserve the user's vocabulary unless local evidence clearly contradicts it.
- Propose an arrangement as `candidate blocks + candidate edges + evidence`,
  rather than asking broad blank questions such as "what are the modules?".
- Ask only about the uncertain decisions that cannot be recovered from files.
- Keep a short review ledger when a block is Codex-drafted or pending approval.

Use this candidate-first format before drawing:

| Candidate | Role | Evidence | Proposed edges | Open decision |
|---|---|---|---|---|

This is especially important for early projects. A repo may already have the
right conceptual architecture even when only part of the implementation exists.

## Layer Topology Is Not Always A Pipeline

Do not assume numbered layers imply a serial dataflow. Classify the topology
before drawing arrows:

- **Pipeline:** data moves through each layer in order, and each layer mutates or
  transforms the artifact for the next layer.
- **Hub-and-spoke:** one layer is the source of truth or control plane, while
  other layers enrich, read, or review it.
- **Read-only context plane:** a layer provides prices, metadata, embeddings,
  policy, or model context, but does not own canonical facts.
- **Feedback loop:** a UI, review, or decision layer writes confirmed results
  back to the canonical layer.
- **Taxonomy:** the layers are responsibility groups, not edges.

Draw the strongest factual flow, then add lighter side-service, read-only, or
feedback edges. If a layer is a context provider, label it as read-only and keep
it off the canonical write path. If a layer is an interaction surface, check
whether it only reads, or whether human-confirmed decisions/journal entries
write back to the core.

## Macro-First Construction

Build the top-level diagram from large architecture questions first:

1. What is the system for?
2. What enters the system, what durable state does it create, and who consumes it?
3. What are the 4-7 macro blocks or work divisions a maintainer would actually
   hand off, review, secure, or deploy?
4. Which edges between those macro blocks carry the important contracts?
5. Which macro edge is still too opaque and deserves local decomposition?

Do not assemble architecture from local component clusters. Start with the
system boundary, macro blocks, and macro edges; then refine downward only where
the macro story is unclear.

A work-division block is valid when it has handoff meaning: independent
responsibility, input/output contract, review criteria, ownership, state
mutation, or privacy/security consequences. File or folder names may appear as
evidence inside that block, but they are not the reason the block exists.

## Node Eligibility

A main node should satisfy most of these:

- It owns a stable responsibility that can be named without mentioning a folder.
- It has an input/output contract, state mutation, data artifact, API boundary,
  user decision point, or external dependency.
- It explains a review question the maintainer cares about: safety, data flow,
  write path, model quality, privacy, scheduling, or product handoff.
- It can be backed by repository evidence in `signature`, `note`, `BLUEPRINT.md`,
  or adjacent narrative.

Keep implementation evidence inside the node:

- Use `signature` for functions, scripts, routes, SQL tables, task names, or API
  methods.
- Use `note` for files, JSON/CSV/SQLite artifacts, key guards, and caveats.
- Use `insets` for a small internal mini-map when a macro block needs 2-5 visible
  substeps but those substeps should not become same-weight top-level nodes.
- Use `expand` for a drill-down diagram when internals deserve their own view.

## What To Keep Out Of Main Nodes

- Folders, packages, React components, or class names that do not define a
  boundary by themselves.
- API keys, config flags, prompt files, MCP servers, and other side services that
  only constrain a block. Put these in `callouts` unless the whole review is
  about that integration.
- Planned roadmap items, unless they are part of the approved product contract or
  architecture question. When included, use `style: dashed` and a note such as
  `planned consumer` or `calibration boundary`.
- Every intermediate helper in a pipeline. Collapse helpers into one card when
  the external contract is the same.

## Diagram Composition

Translate the semantic graph into a small composition plan before assigning
coordinates:

1. Mark the strongest end-to-end path as `primary`.
2. Mark read-only inputs and policy/model providers as `context`.
3. Mark supporting flows as `secondary` and human/write-back loops as `feedback`.
4. Add only the `left` / `center` / `right` lanes needed to preserve the reading
   order across layers.
5. Let the renderer calculate coordinates; use explicit `layout` only for local
   project refinement after the composition plan is accepted.

The same path and lane plan should feed both deterministic rendering and the
generated-image prompt. This keeps visual exploration flexible without asking
either track to rediscover the architecture narrative from raw edges.

- Aim for about 8-12 primary nodes in the top-level view. If the truthful diagram
  needs more than roughly 14, split the view or add drill-down subdiagrams.
- Prefer 4-7 macro blocks for the first conversation with a maintainer. Add
  more only after the macro edges fail to explain the system.
- Use `groups` for semantic regions: entry adapters, external sources, core
  pipeline, storage/products, read/write boundary, frontend/output, operations.
- Do not use groups as folder mirrors. A group should answer "what kind of
  architecture space is this?", not "which directory is this in?".
- Put the main data/workflow chain on the strongest visual path. Orchestration,
  config, model calibration, and side services should be lighter or peripheral.
- Label edges with concrete artifacts or contracts, such as `RawPost`,
  `DailyMetric`, `PATCH`, `resolutions.csv`, or `venue + citations`. Leave
  config/orchestration edges unlabeled unless the label adds meaning.
- Prefer callout chips for constraints, runtime knobs, manual fallbacks, and
  external side services that would clutter the primary map.
- Use card insets or drill-down subdiagrams when the macro component is truthful
  but visually under-explained.

## Change / Focus Overlays

Architecture diagrams often need to support code review or local design
discussion. Do not redraw the whole architecture just because the current change
touches one area.

Use this pattern instead:

1. Keep the canonical macro graph stable.
2. Highlight the affected macro block or region.
3. Dim unrelated regions if it helps scanning.
4. Add A/B/C markers for touched blocks.
5. Open a subview inside the affected block when the local implementation needs
   detail.

For example, a frontend-only change in a travel map should mark the read-side
framework or expand `useTravelData -> useTripFilters -> Map props ->
Presentation`, while leaving the write-side curation and canonical data plane
unchanged. Promote the change back to the macro graph only if it changes the
data contract, access boundary, durable state, deployment boundary, or external
dependency.

## Lessons From Approved References

### Paper Resolver / Bibliography Enrichment

The useful abstraction is not the package layout. It is:

- entry adapters: CLI and Web UI;
- external venue sources: Semantic Scholar and DBLP;
- core resolver/ranking/proposal pipeline;
- external record-system read/write boundary and human review/report products.

It is acceptable to merge two implementation write paths into a high-level
write-back boundary when the architecture question is "what owns mutation into
the external record system?". Show the guard, review, and idempotent write
contract inside that card instead of splitting by source file.

### Social Sentiment Monitor

The useful abstraction is the product/data contract chain:

`watchlist -> scraper -> sentiment engine -> aggregator -> DailyMetric -> dashboard/notify/backtest`

Dashed M4/M5/M7-style nodes are acceptable when the approved project contract
includes planned evaluation, backtest, or notification boundaries. Make planned
status visible and keep the main observed data path stronger than the roadmap
path.

## Failure Flags

Flag a diagram when you see:

- Fake folder grouping: top-level boxes are mostly directories or component names.
- Over-detailed internals: helper functions crowd out architecture contracts.
- Missing contracts: nodes have names but no inputs, outputs, artifacts, or guard
  conditions.
- GT overfit: the diagram matches an example's visual style while ignoring repo
  evidence.
- Unreadable mega-diagram: too many same-weight cards, edge labels, or crossing
  arrows for a single view.

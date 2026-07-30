---
name: methodology-page-builder
description: Build, rewrite, audit, or update an evidence-backed methodology page for a research-facing website or ML product. Use when work involves method explanations, architecture diagrams, model-comparison evidence, performance tables, citations, limitations, or keeping a public methods page synchronized with deployed artifacts. Do not use for ordinary landing-page copy or unrelated frontend changes.
---

# Methodology Page Builder

Build the shortest defensible explanation of the deployed product. Keep research facts, architecture truth, and presentation outputs synchronized.

## Required workflow

1. Inspect the repository instructions, current methodology page, implementation, tests, research notes, architecture schema, and frozen evaluation artifacts.
2. Read `docs/methodology_page_workflow.md` completely. Treat it as the detailed writing, evidence, visual, implementation, and validation standard.
3. State any material ambiguity before editing. Never silently choose between conflicting model revisions, splits, metrics, or deployment facts.
4. Establish the authoritative source for every changed claim:
   - topology: architecture schema;
   - evaluation: frozen comparable metrics;
   - deployment: production artifact metadata;
   - prose: reviewed research notes plus current implementation;
   - presentation behavior: stylesheet, renderer, and tests.
5. Build or update an evidence ledger before drafting quantitative claims.
6. Structure the page around task, architecture, data discipline, method comparison, calibration or decision policy, current record, limitations, and references.
7. Update only the affected source-of-truth layer. Regenerate derived diagrams instead of editing generated assets manually.
8. Keep mechanism, evidence, interpretation, and limitation statements distinct.
9. Run the evidence, content, visual, and regression gates defined in the workflow.
10. Report changed files and any unresolved evidence gaps. Do not report routine successful steps.

## Hard requirements

- Use only claims supported by current code, frozen artifacts, or relevant primary research.
- Compare methods only on identical evaluation units, splits, candidate pools, and revisions.
- Show uncertainty and distinguish the numerical leader from the deployed method.
- Do not call similarity a probability, accuracy, confidence, or percentage match unless the corresponding calibration supports that meaning.
- Keep primary ranking signals visually and statistically separate from context-only or explanation signals.
- Preserve a valid non-match when the product is open-set.
- Keep limitations concise and decision-relevant.
- Never expose commit hashes, local paths, notebook-part labels, debugging history, abandoned implementation details, or personal information.
- Never report author counts on the methodology page.
- Do not rewrite unrelated page content or styling.
- Do not treat DOM tests as sufficient visual validation.

## Output

Deliver the implemented page or audit requested by the user, plus:

- the authoritative evidence used for each material quantitative change;
- the files changed;
- failed or unavailable validation checks;
- unresolved discrepancies that prevent a defensible public claim.

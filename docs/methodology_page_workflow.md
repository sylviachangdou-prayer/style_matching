# Methodology Page Build Workflow

Use this workflow for research-facing product pages that must explain a model, justify its design, and remain visually legible without overstating the evidence.

## 1. Define the page contract

Before writing or designing, record:

- **Audience:** an informed user, reviewer, collaborator, or client.
- **Decision explained:** what the system computes, why this method was selected, and what the result means.
- **Claims allowed:** only claims supported by the current code, frozen artifacts, or cited research.
- **Claims excluded:** unpublished aspirations, stale experiments, implementation history, personal information, commit hashes, and internal troubleshooting.
- **Required restraint:** limitations remain visible, but ordinary prose should not dramatize weaknesses or expose abandoned implementation details.

The methodology page is not a paper, model card, changelog, or marketing page. It is the shortest defensible explanation of the deployed product.

## 2. Establish sources of truth

Do not edit the same fact independently in several presentation files.

| Content | Authoritative source | Presentation output |
|---|---|---|
| System topology | `docs/architecture/schema.yaml` | `method-flow.js`, SVG/HTML/PNG diagrams |
| Evaluation numbers | frozen metrics in `docs/` or versioned artifacts | performance figure and prose |
| Deployed model and index facts | production artifact metadata | performance record |
| Method narrative | reviewed research notes and current implementation | `web/static/method.html` |
| Visual rules | shared site design plus method stylesheet | `site-shell.css`, `method-flow.css` |
| Behavioral requirements | tests | `web/tests/test_method_ui.py` |

If two sources disagree, stop and resolve the deployed fact before editing the page.

## 3. Build an evidence ledger

Create a compact working table before drafting:

| Claim | Evidence | Scope | Page destination | Status |
|---|---|---|---|---|
| What the primary model measures | paper + implementation | deployed system | architecture/method | verified |
| Why a model was selected | locked held-out metrics + selection rule | frozen comparison | performance section | verified |
| What a score means | API contract + calibration artifact | current version | task/limitations | verified |
| Runtime or corpus fact | production metadata | current artifact only | performance record | verified |

Every quantitative statement must identify its evaluation scope. Never combine numbers from different splits, candidate pools, model revisions, or artifact versions in one comparison.

Use these labels internally:

- **Verified:** may be stated directly.
- **Experimental:** must be labelled experimental.
- **Unavailable:** omit the result or state that it is unavailable.
- **Planned:** keep out of the methodology page unless the page explicitly includes a roadmap.

## 4. Design the information hierarchy

Use the following default order:

1. **Task definition** — what problem is being solved and what a valid non-match means.
2. **System architecture** — the main inference path and strictly separate supporting signals.
3. **Data discipline** — source independence, language, rights, and leakage controls.
4. **Method comparison** — alternatives, locked metrics, uncertainty, and the selection rule.
5. **Calibration or decision policy** — how raw similarity becomes a status or rejection.
6. **Current system record** — deployed model, retrieval method, coverage dimensions, and latency.
7. **Known limitations** — concise boundaries that affect interpretation.
8. **References** — only works actually used to support the page.

Do not report author counts on the methodology page. Author-library size, indexed profile coverage, and formally evaluable coverage are different quantities and belong in the data card, API metadata, or Author Library.

## 5. Write the methodology

### 5.1 Draft claims before prose

For each section, write one sentence answering each applicable question:

- What does this component do?
- What evidence enters it?
- What is physically or statistically excluded?
- Why was it chosen over alternatives?
- What can the user infer from its output?

Then combine only the necessary sentences into prose.

### 5.2 Use precise terminology

Prefer:

- “ranked retrieval” over “AI identification”;
- “cosine similarity” over “confidence” unless calibration exists;
- “source-heldout” over “unseen” when only sources were held out;
- “original-language primary text” over “authentic text”;
- “numerical leader” and “deployed method” when they differ;
- “did not pass the pre-declared selection gate” over “failed.”

Do not call a similarity score a probability, accuracy, or percentage match.

### 5.3 Separate four kinds of statements

- **Mechanism:** what the deployed system computes.
- **Evidence:** what the frozen evaluation showed.
- **Interpretation:** what the user may conclude.
- **Limitation:** where that conclusion stops.

Do not mix these into one promotional sentence.

### 5.4 Cite selectively

Cite research only when it supports a design choice, evaluation protocol, or interpretation boundary. A reference list is not evidence unless the relevant claim appears in the prose. Prefer primary papers and official model documentation.

## 6. Build the architecture view

Treat `docs/architecture/schema.yaml` as the architecture source of truth.

1. Identify the primary ranking path.
2. Identify context-only, memory, calibration, and explanation planes.
3. Keep supporting planes visually separate from the ranking path.
4. Use stable component IDs and specific edge labels.
5. Put implementation details inside card notes, not as top-level nodes.
6. Keep the diagram compact; use a regular grid when it improves scanning.
7. Avoid crossing arrows, animated arrows, and decorative motion.
8. Short labels must remain subordinate to body text.
9. Mark experimental or evidence-gated components without making status labels visually dominant.
10. Regenerate all derived diagram assets after a schema change.

Required maintenance chain:

```text
docs/architecture/schema.yaml
  -> deterministic renderer
  -> diagram.svg / diagram.html / diagram.render.png
  -> web/static/method-flow.js
  -> web/static method view
```

Keep Mermaid only as a diff-friendly fallback. Do not use a manually edited screenshot as architecture truth.

## 7. Present method comparisons

Use a compact figure for the primary retrieval metric and a two-column table for method definitions.

The figure must show:

- the metric name;
- point estimates;
- uncertainty intervals;
- the numerical leader;
- the deployed method when different.

The method table must explain each candidate as a complete methodological design, not a stack of keywords. State its representation, training or adaptation status, aggregation rule, and purpose in the comparison.

Selection prose must report:

1. the pre-declared decision rule;
2. the observed difference;
3. its paired uncertainty interval;
4. any subgroup or calibration gate;
5. the resulting deployment decision.

Do not select a more complicated method because its point estimate is marginally higher.

## 8. Apply the visual system

The methodology page should inherit the product’s navigation, typography, spacing, background treatment, and color tokens. It may be denser than the home page, but it should not become a paper pasted into a browser.

Requirements:

- body text remains the dominant reference size;
- highlights preserve readable contrast;
- tables use explicit column widths and comfortable row height;
- method cards may vary in height to fit their content;
- color distinguishes semantic roles, not arbitrary steps;
- diagrams have no unrelated background image;
- the page remains usable without JavaScript through a static fallback;
- mobile layouts preserve reading order rather than shrinking the desktop diagram.

## 9. Implement surgically

Touch only the relevant layer:

- change a system fact in the authoritative metadata or schema first;
- change narrative wording in `method.html`;
- change diagram content in `schema.yaml`, then regenerate;
- change layout or typography in `method-flow.css`;
- change schema-to-DOM rendering in `method-static.js`;
- update tests only when the intended contract changes.

Do not manually patch generated `method-flow.js`, SVG, or PNG assets.

## 10. Validate

### Evidence gate

- Every number matches a frozen artifact.
- Compared methods use the same split and evaluation unit.
- Deployed model wording matches the production index.
- Uncalibrated scores are not described as probabilities.
- Experimental functions are labelled accurately.
- References support claims made on the page.

### Content gate

- The page explains the task, architecture, data discipline, selection, decision policy, current record, and limitations.
- No author count appears.
- No commit hash, local path, personal detail, debugging history, or notebook-part label appears.
- No stale method or abandoned experiment is presented as deployed.
- Repetition and generic promotional language have been removed.

### Visual gate

- Diagram text is not clipped.
- Arrows do not cross.
- Tables are readable at desktop and mobile widths.
- Highlights and table text pass contrast inspection.
- Static fallback works with JavaScript disabled.
- Navigation and shared page atmosphere match the rest of the product.

### Regression gate

Run:

```bash
python3 -m pytest web/tests/test_method_ui.py
python3 -m pytest web/tests/test_web_api.py
```

Also inspect `method.html` in a browser at desktop and mobile widths. Passing DOM tests is necessary but not sufficient for visual acceptance.

## 11. Update an existing page

For any later model, data, or design change:

1. Identify which claim changed.
2. Locate its authoritative source.
3. Update the evidence ledger.
4. Edit the source of truth.
5. Regenerate derived architecture assets if required.
6. Update only affected prose, figure values, or record rows.
7. Run evidence, content, visual, and regression gates.
8. Record the artifact version outside the public prose when reproducibility requires it.

Do not rewrite the entire page for a local change.

## 12. Definition of done

The methodology page is complete when a critical reader can answer:

- What exactly is ranked?
- Which signal controls the ranking?
- Which signals are contextual only?
- How were leakage and source dependence controlled?
- What alternatives were compared on equivalent evidence?
- Why is the deployed method justified?
- What do the scores permit the user to conclude?
- When should the system return no match?
- What remains experimental or unsupported?

If any answer requires private knowledge from the developers, the page is not finished.

# StyleMatch web app

Everything web-related lives in this folder; model/corpus code stays in `scripts/`.
The backend only imports `scripts.multilingual_style_index.StyleIndex` and loads the
prebuilt index once at process start — no model downloads or corpus encoding inside a
request handler.

```
web/
  api/        FastAPI backend (match endpoint, language detection, explanations, demo fixture)
  static/     framework-free frontend (deploy to Vercel as-is)
  config/     web-layer thresholds (weights.yaml)
  tests/      API tests that run without any model download (demo mode)
  Dockerfile  Hugging Face Space (Docker SDK) image
```

## Local run

```bash
./web/preview.sh
# Home:   http://127.0.0.1:8000
# Method: http://127.0.0.1:8000/method.html
# Authors: http://127.0.0.1:8000/authors.html
```

The preview script uses the local challenger index and an already working Python
installation; it does not resolve or reinstall the training stack. Set
`STYLEMATCH_INDEX_DIR` only when testing another index.

The app looks for `artifacts/multilingual_style_index_challenger_v1/` first (the index
built from the model selected in `artifacts/model_comparison_v1.json`), then falls back
to the retired `artifacts/multilingual_style_index_v1/` baseline index. When neither is
present it serves a clearly flagged **demo fixture** (`"demo": true`, banner in the UI)
so frontend work can proceed while the Colab training run finishes. Set
`STYLEMATCH_INDEX_DIR` to override the search entirely.

Tests: `python -m pytest web/tests -q` (runs in demo mode, no downloads).

## Deploy: backend on Hugging Face Spaces (free CPU)

1. Upload the Colab artifacts to a HF dataset repo, e.g.:

   ```bash
   huggingface-cli upload YOUR-USERNAME/stylematch-index \
     artifacts/multilingual_style_index_v1 . --repo-type dataset
   ```

   If the index was built with the fine-tuned encoder, upload
   `artifacts/mstyledistance_stylematch_v1` as a model repo and make sure
   `metadata.json`'s `model_name` points at it (or a hub id).

2. Create a **Docker SDK** Space, then push this repo with the Dockerfile at the
   Space root (HF requires that):

   ```bash
   cp web/Dockerfile Dockerfile
   git push space main
   ```

3. In Space settings → Variables set:
   - `STYLEMATCH_HUB_REPO=YOUR-USERNAME/stylematch-index` (downloaded at startup)
   - `STYLEMATCH_CORS_ORIGINS=https://your-app.vercel.app` (comma-separated)

The Space serves both the API and the static page, so the Space URL alone is a
working deployment; Vercel is the nicer public front door.

## Deploy: frontend on Vercel (free)

`web/static/` is plain HTML/JS — no build step. Edit `web/static/config.js`:

```js
window.STYLEMATCH_API = "https://YOUR-USERNAME-stylematch.hf.space";
```

then `vercel deploy web/static` (or import the repo in Vercel with `web/static`
as the root directory).

## Environment variables

| Variable | Default | Meaning |
| --- | --- | --- |
| `STYLEMATCH_INDEX_DIR` | challenger index, then baseline index | Prebuilt index directory (overrides the search order) |
| `STYLEMATCH_HUB_REPO` | unset | HF repo to download the index from when the dir is missing |
| `STYLEMATCH_HUB_REPO_TYPE` | `dataset` | Repo type for the download |
| `STYLEMATCH_DEVICE` | `auto` | `cpu`, `cuda`, `mps`, or `auto` |
| `STYLEMATCH_CORS_ORIGINS` | `*` | Comma-separated allowed origins |

## Honesty contract enforced by this layer

- Style and topic sub-scores are always returned separately alongside the composite.
- Literary and rhetorical passages are pooled into one author-language profile; corpus provenance
  remains in the API metadata but is not shown as a second user-facing author label.
- Cross-language responses carry `confidence: "reduced"` and are ranked per target
  language (uncalibrated scores are not comparable across languages); the UI says so.
- Below `affinity.low_confidence_threshold` (web/config/weights.yaml) the UI leads
  with an explicit low-confidence state instead of a fake-precise label.
- Demo mode is impossible to mistake for a measurement: flagged in the payload,
  `score_status: demo_fixture`, and a banner in the UI.
- Responses include score/artifact versions, profile strategy, author profile text, style traits,
  and decade status. Decade results remain absent until support and author-heldout gates pass.
- Application logs contain only language, mode, latency, confidence state, returned labels, and
  decade label. User text is never written to monitoring logs.

## Frontend extras

- **Permanent navigation**: Home, Method, and the full Author Library are direct
  routes. The library is generated from `data/source_registry/all_people.csv` by
  `scripts/export_author_library.py` and provides search plus language/register filters.
- **Distinct results gallery**: successful matching replaces the entry portrait with four
  local public-domain museum strips. Selecting a result changes the emphasized artwork;
  the closest original-language passage is always visible.
- **Static method map**: `method.html` renders the schema-backed primary path as a
  non-crossing diagram with separate context, profile-memory, and evidence planes.
- **Test a new passage / Clear**: after a match the results end with a "Test a new
  passage" button (and the counter row gains a "Clear" button) that resets the input
  and results in one click.
- **Export**: "Export as PDF" opens the browser print dialog on a dedicated white
  result sheet (input passage, all scores, evidence, representative passages in navy,
  the site URL, and a QR code); "Save as image" renders the same sheet to a PNG.
  The PNG renderer (`html2canvas`) and QR generator (`qrcode-generator`) are lazy-loaded
  from jsDelivr only when an export button is pressed; if the CDN is unreachable the
  PDF path still works and the QR silently degrades to the plain URL. Author portraits
  are intentionally excluded from exports so cross-origin images can never taint the
  canvas. Set `window.STYLEMATCH_SHARE_URL` in `config.js` to control the URL/QR
  printed on exports.
- The page footer carries the system abstract (open-set retrieval, dual-channel
  architecture, uncalibrated-score and cross-language caveats) — keep it in sync with
  the honesty contract above if scoring semantics change.

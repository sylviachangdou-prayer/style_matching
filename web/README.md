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

The preview script uses the local expanded coverage-beta index and an already working Python
installation; it does not resolve or reinstall the training stack. Set
`STYLEMATCH_INDEX_DIR` only when testing another index.

The app looks for `artifacts/multilingual_style_index_gutenberg_v3/` first, then the
earlier challenger index, then the retired baseline. The expanded index keeps the
selected encoder but broadens candidate coverage; its metadata labels it a
coverage-first beta because shared-candidate ranking did not improve. When no index is
present the app serves a clearly flagged **demo fixture** (`"demo": true`, banner in
the UI). Set `STYLEMATCH_INDEX_DIR` to override the search entirely.

Tests: `python -m pytest web/tests -q` (runs in demo mode, no downloads).

## Deploy: backend on Hugging Face Spaces (free CPU)

Free Docker Space = 2 vCPU / 16 GB RAM / 50 GB disk. Enough for this stack; the
constraint is sleep, not size (see "Sleep and cold start" below).

Serving needs three things, not the whole `artifacts/` tree:

| Piece | Size | Where it must live |
| --- | --- | --- |
| Index `multilingual_style_index_gutenberg_v3` | 13 MB | HF **dataset** repo, pulled at startup |
| Fine-tuned encoder `multilingual_author_style_v1` | 2.1 GB | HF **model** repo |
| `intfloat/multilingual-e5-base` (topic channel) | ~1.1 GB | pulled from the Hub automatically |

1. Upload the index and the encoder:

   ```bash
   huggingface-cli upload YOUR-USERNAME/stylematch-index \
     artifacts/multilingual_style_index_gutenberg_v3 . --repo-type dataset

   huggingface-cli upload YOUR-USERNAME/multilingual-author-style-v1 \
     artifacts/multilingual_author_style_v1 . --repo-type model
   ```

2. **Repoint `model_name` before uploading the index.** `metadata.json` currently
   says `"model_name": "artifacts/multilingual_author_style_v1"` — a local path
   that does not exist inside the Space. `load_index()` builds `StyleIndex`
   unguarded inside `lifespan`, so this raises and **the Space fails to boot**;
   the demo fallback only covers a missing index, not an unloadable encoder.
   Change it to the hub id:

   ```json
   "model_name": "YOUR-USERNAME/multilingual-author-style-v1"
   ```

3. Create a **Docker SDK** Space and push this repo with the Dockerfile at the
   Space root (HF requires it there):

   ```bash
   cp web/Dockerfile Dockerfile
   git push space main
   ```

4. Space settings → Variables:
   - `STYLEMATCH_HUB_REPO=YOUR-USERNAME/stylematch-index`
   - `STYLEMATCH_CORS_ORIGINS=https://your-frontend-domain` (comma-separated)

The Space serves both the API and the static page, so the Space URL on its own is
a working deployment.

### Sleep and cold start

A free Space sleeps after ~48 h without traffic. The next visitor's request wakes
it; the Docker **image** stays cached, so this is a container restart, not a rebuild.

What that restart costs depends on where the weights are:

- **Weights downloaded at startup (current behaviour)**: the encoder (2.15 GB) and
  the topic model `intfloat/multilingual-e5-base` (~1.15 GB of the repo's 5.3 GB —
  sentence-transformers pulls `model.safetensors`, not the onnx/openvino variants)
  are fetched into `HF_HOME` on every wake, because free Spaces do not persist
  writes outside the image. ~3.3 GB per cold start. How long that takes depends on
  Hub-to-Space bandwidth, which is not documented — measure it once rather than
  trusting an estimate.
- **Weights baked into the image**: cold start is container start plus loading the
  model into RAM — roughly 30–60 s on CPU.

To bake them in, add this to the Dockerfile after the pip install, once the model
repo exists:

```dockerfile
ARG STYLEMATCH_MODEL_REPO=YOUR-USERNAME/multilingual-author-style-v1
RUN python -c "from huggingface_hub import snapshot_download; \
    snapshot_download('${STYLEMATCH_MODEL_REPO}'); \
    snapshot_download('intfloat/multilingual-e5-base')"
```

The image gets ~3.2 GB bigger and builds slower, but every wake after that is fast.
The 12 MB index can stay a startup download.

To avoid sleep entirely you need a paid always-on Space. The free workaround is
`.github/workflows/keep-warm.yml` in this repo, which pings `/api/health` on a
schedule.

It ships **off**. The scheduled run exits immediately unless the repository
variable `KEEP_WARM` is `on`, so committing it does not start anything.

Set two repository variables under **Settings → Secrets and variables → Actions →
Variables**:

| Variable | Value |
| --- | --- |
| `SPACE_URL` | `https://YOUR-USERNAME-stylematch.hf.space` |
| `KEEP_WARM` | `on` to keep it awake, `off` (or delete) to let it sleep |

Day to day:

- **Turn it on before sharing the link** — set `KEEP_WARM=on`. Takes one edit in the
  GitHub UI, no commit.
- **Turn it off afterwards** — set it to `off`. The workflow stays installed and idle.
- **Wake it once, right now** — Actions → *Keep Space warm* → **Run workflow**. A
  manual run ignores `KEEP_WARM`, so this works even while keep-warm is off. Useful
  a minute before you send someone the link.

The workflow checks a UTC schedule every 12 hours but sends a request only in every
third half-day slot, so successful scheduled pings are 36 hours apart. POSIX cron
cannot express a rolling 36-hour interval directly. The 17-minute offset avoids the
most congested start-of-hour window.

Ping frequency does not change what this costs. If the pings succeed at preventing
sleep, the container is resident 24/7 either way. The interval only buys safety
margin, and the pings themselves are a few KB. A 36-hour interval leaves roughly 12
hours before the documented 48-hour sleep threshold.

So do not stretch the interval to save anything; stretch it only if you accept the
risk. Two reasons a near-48 h cron misses:

- GitHub's scheduled workflows are best-effort. They are delayed under load
  (on-the-hour schedules are the most congested) and can be skipped outright. A cron
  set at the deadline will sometimes fire after the Space has already slept.
- In a public repo, GitHub disables scheduled workflows after ~60 days with no
  repository activity. Keep-warm then stops silently.

The real lever is whether keep-warm is on at all. With the weights baked into the
image, leaving it off costs the first visitor 30–60 s rather than minutes, which is
usually the right trade for a personal site.

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
| `STYLEMATCH_INDEX_DIR` | expanded index, challenger index, then baseline | Prebuilt index directory (overrides the search order) |
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

The reusable research, writing, architecture, visual, and validation process for
the methodology page is documented in
[`docs/methodology_page_workflow.md`](../docs/methodology_page_workflow.md).

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

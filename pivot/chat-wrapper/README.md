# chat-wrapper — photo analysis disguised as chat

Standalone page + server. A user drops a photo in chat; the assistant
(Muse Spark with tool-calling when `.cache/meta-api-key.txt` exists, else
scripted) calls `analyze_photo`, and the analysis streams transparently as a
tool card: stages, logs, artifacts inline, summary with numbers.

## Run

```powershell
. .\scripts\env.ps1
.\.venv\Scripts\python.exe pivot\chat-wrapper\chat_server.py --port 8150
# open http://127.0.0.1:8150/
```

Deep link: `#run=<run_id>` replays `runs/<id>/events.jsonl` (static, no recompute).
Health: `/api/health` (muse live/replay, active_run, usage).

## What it reuses (no duplication)

`analyze.py` imports `pivot/sam3dbody-ingest/run_all.py` helpers directly
(`load_pose/load_dense/load_3db`, `pose_run/person_box/seg/normal/depth`,
`fit_3db/identity_refine/mesh_and_landmarks/project_cam`, render/overlay
helpers) + `joint_solve.solve_joint`. One-slot `ModelSlot` keeps a single
model resident (`del + gc + empty_cache` between stages, 95% cap at import).

Stages (data-driven, mirrored in `STAGE_META` + `analyze.STAGES`):
`vision (Muse Spark structured photo scan + person pick) → ingest → pose → maps (0.4b seg/normal/depth) → body (3DB body-fit + refine + joint + sheets) → report (stats.json)`.
1b pointmap/matting via the maps-gate `full_maps` pick or the upload checkbox.

## First message: photo → Muse Spark structured object

The first message uploads the photo to `muse-spark-1.3-contributor`
(`POST https://api.meta.ai/v1/responses`, Responses API, `reasoning.effort=low`,
`store=false`, strict `json_schema`) and asks for:

```json
{"photo-description": "one sentence", "people-detected": [{"bbox": {"x_min","y_min","x_max","y_max (0-1000 rel)"}, "description": "one-line appearance note"}]}
```

Boxes map to pixels, are drawn numbered (`vision_boxes.jpg`) and gated:
**1 person** → confirm box / Sapiens-auto / cancel; **N people** → one
option-card per person (`Person 1 — <desc>`…); **0** → auto / cancel.
The chosen box becomes the fit's manual `bbox`. Scan failure degrades loudly
to Sapiens auto (warn + `vision: unavailable`), never silently. The parsed
scan is saved as `vision_scan.json` (provenance `live`).

## Every API call displayed (full POST + full return)

* In-run Meta calls stream as `api.call` (model, URL, purpose, **complete POST
  body**) + `api.return` (HTTP status, latency, request id, usage, **complete
  response JSON**) events, rendered as amber inspector cards with collapsible
  raw JSON. Inline base64 images are replaced by `<N base64 chars redacted>`;
  the key travels in the header only and is never emitted, logged, or written.
* Text-only chat replies carry the same pair as `api_trace` in the
  `/api/chat` response + chatlog entry, rendered under the assistant bubble
  and in replayed history.

## Debug actions (standalone cards, not gated)

A debug strip above the composer offers two actions on the attached photo.
Both persist to `chats/<id>/debug/<n>/` and to the chatlog, and both display
the full POST + full return.

* **🔍 Describe this image** (`muse-spark-1.3-contributor`, free text):
  `POST /api/debug/describe {chat_id, image_b64, filename, prompt?}`.
* **🧩 Detect People** (`sam-3.1`, concept noun phrase, default `person`):
  `POST /api/debug/detect {chat_id, image_b64, filename, concept?}`. One
  request per concept (API rule — no comma lists). `metadata.mask_encoding:
  one_bit` for small payloads. The special-token lane is parsed in-house
  (`_parse_sam_line`), masks decoded with the vendored zero-dep
  `meta_sam_parser` (`_vendor/`, box-local rasters pasted at their box
  origin), rendered as one card per mask (tint + box + id, click to zoom)
  plus a combined overlay. Zero matches is a normal empty result, not an
  error — the card says to try a plainer noun phrase.
* **Describe this mask →** on any mask card crops to the box (15 % pad) with
  the background dimmed and asks Spark to describe the highlight:
  `POST /api/debug/describe-mask {chat_id, debug_id, mask_id}`.
* Files served via `GET /api/chats/<id>/file/debug/<n>/<name>` (traversal-guarded).

## Sapiens debug actions (local models, same code as the pipeline)

Four more debug cards call the exact `run_all.py` stages standalone — one
model resident at a time, evicted after each call. Blocked with 409 while a
body-scan run holds the GPU.

* `POST /api/debug/sapiens-pose` → 308-keypoint overlay + auto box + scores.
* `POST /api/debug/sapiens-seg` → segmentation overlay + class counts + skin pixels.
* `POST /api/debug/sapiens-normal` → surface-normal map (runs seg first for the skin mask, evicted between).
* `POST /api/debug/sapiens-depth` → pointmap depth in metres + range (same seg-first pattern).
* `POST /api/debug/body-fit` → full local chain (pose → maps → 3DB fit → identity refine → joint solve → 06/07/08/09 sheets), returns mesh URL + `identity_joint.json` + `fit_stats.json`.
* `POST /api/debug/export` → packages a fitted OBJ into `avatar_character.glb` + `export_stats.json` (CPU, trimesh), collecting sibling spec files.

Concurrent GPU debug calls get `409` (one at a time via `GPU_LOCK`), same as a held body-scan run.

Verified live on `Emiru.webp`: pose 0.968 median, box `[0, 150.3, 999, 999]`;
9 seg classes; depth `[0.342, 0.519]` m — all matching the pipeline's own runs.

Verified live on `Emiru.webp`: describe OK; `person` → 1 mask, 722,087 px,
pixel-clean hair/face boundary; mask-crop describe OK (all HTTP 200).

## Chatlogs — every turn saved, listable anytime

* Every `POST /api/chat` appends to `chats/<chat_id>/chat.jsonl` (user text +
  assistant text + tool_call + run link); gate clicks append as
  `[<run> choice] <id>` user entries. A reload never loses history.
* `GET /api/chats` → `[{chat_id, updated_at, n_messages, preview}]` (sidebar).
  `GET /api/chats/<id>` → full message list. The page keeps `chat_id` in
  `localStorage`, has a `+ New chat` button, and each assistant entry with a
  `run_id` gets a `▶ replay run` link (streams the persisted `events.jsonl`).
* `chats/` is gitignored like `runs/`.

## Standard flow — gated at every step (option cards)

The run pauses after each stage and waits for a clickable pick; the pipeline
never advances on its own (`Run.open_gate` blocks `execute()` on
`gate_event`; `POST /api/runs/<id>/choice {choice}` resumes; cancel sets the
cancel flag and maps to `run.end{cancelled}`, not failure):

| after | prompt | option cards |
| :--- | :--- | :--- |
| ingest | photo received (WxH), find the person? | Continue · auto-detect box (Recommended) / Cancel |
| pose | 308 kpts (score), box — continue to maps? | Continue · seg/normal/depth (Recommended) / Cancel |
| maps | base maps done (depth range) — 3D fit or slow 1b first? | Continue · 3D fit (Recommended) / Run full_maps first / Cancel |
| body | fit done (refine path, \|β\|) — write report? | Continue · write report (Recommended) / Cancel |

Gates stream as `gate.open` SSE (amber card with buttons) and resolve as
`gate.answer` (persisted in `events.jsonl`, so replays show the pick).
Headless callers set `opts.auto_continue=True` to take defaults without
blocking (`smoke_photo.py` does this).

## Endpoints

* `POST /api/chat {chat_id?, text, image_b64?, filename?, full_maps?}` → `{chat_id, run_id?, assistant_text, tool_call, events_url}`; `409 {active_run}` while busy (one run at a time — in-process models + 95% cap make concurrency OOM-unsafe).
* `GET /api/chats` (chatlog list), `GET /api/chats/<id>` (full history).
* `GET /api/runs/<id>/events` (SSE + `Last-Event-ID`; replays of finished runs close cleanly via `Connection: close`), `GET /api/runs/<id>/run.json`, `GET /api/runs/<id>/artifact/<aid>`, `GET /api/runs`, `POST /api/runs/<id>/choice {choice}`, `POST /api/runs/<id>/cancel`, `GET /api/health`.
* Busy policy: a new photo while a run **computes** → 409 with a watch/cancel hint. While a run is **paused at a gate** (waiting for your pick), the new photo retires it (`superseded`) and starts — no more stuck Busy. Cancel of a gated run answers its gate, so it ends exactly once.
* Text-call rail: `CHAT_MUSE_MAX_TEXT_CALLS` (default 200, was 12) counts live Muse turns in `usage.json`; over the rail the server says so once and continues in replay mode. Delete `usage.json` to reset the counter.
* `POST /api/debug/describe|detect|describe-mask|spark|muse|sapiens-*|body-fit|export`, `GET /api/chats`, `GET /api/chats/<id>`, `GET /api/chats/<id>/file/debug/<n>/<name>`.

## Transparency contract

* One tool card per run (not one bubble per stage): header `run_all · <id>` + status pill + elapsed + copy-link + cancel; stage rail S1..S6 with dots, thumbnails, metric captions; log tail (last 4 lines); API inspector cards; `<details> raw` with tool JSON + artifact records. The card is display-only — it never contains an answer control.
* **No nested answering:** a gate never renders inside the tool card. It pops up as its own `⏸ Your pick needed` bubble below the card, with one button per option. Results do the same: the run summary carries Download buttons, each mask/Sapiens image gets its own bubble with its action button, replay/watch/cancel are buttons, never bare links.
* Summary (`M3`) shows only: total s, |β| + refine path, focal + box, pose median, depth range, file links — always "character-creation numbers only".
* Failures degrade loudly (`StageError`: zero-area box, non-finite mesh, missing 3DB keys, joint fallback with `applied:false` + warn). Never a bare spinner: instant echo + S1/S2 thumbs in ~1.5s while 3DB grinds; weighted progress (pose 4 / dense 6 / 3DB 55 / joint 35).

## Files

| file | what |
| :--- | :--- |
| `chat_server.py` | server, Run/SSE/runs/409/Muse-or-scripted |
| `analyze.py` | stage adapter over `run_all` + `joint_solve` |
| `chat.html` | sidebar chat list + M1/M2/M3 UI + gate option cards + debug bar/mask cards + SSE + replay |
| `runs/<id>/` | `source.*`, `input/01`, `sapiens/{02..05 + raw}`, `threedb/{06..09 + identity_joint.json}`, `stats.json`, `events.jsonl`, `run.json` (gitignored) |
| `chats/<id>/` | `chat.jsonl` (every turn), `meta.json` (preview/counts), `debug/<n>/` (sources, masks, overlays, crops) (gitignored) |
| `_vendor/meta_sam_parser/` | vendored zero-dep SAM grammar + mask decoder (gitignored via `pivot/`) |
| `smoke_*.py` | headless repros: photo pipeline, vision scan, debug actions |
| `usage.json` | Muse text-call ledger (gitignored) |

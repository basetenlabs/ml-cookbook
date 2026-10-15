# CUSTOMIZATION.md

Map of "if the user asks for X, edit Y." Both researchers and future-Claude use this
to make narrow, surgical changes without exploring the whole codebase.

## Rule of thumb

If the change is about **what color / size / spacing**, it's in `style.css` (almost certainly a CSS variable in `:root`).

If the change is about **how a rollout is laid out** (turn timeline, prompt cards, etc.), it's in `renderers.js` (one function per format).

If the change is about **what's detected from the run directory**, it's in the skill — re-run the skill to regenerate `run.json`.

Don't edit `dashboard.py` unless you're adding a new endpoint or a new file format. Resist the urge to "improve" it.

## The map

| User says | Edit | Where exactly |
|---|---|---|
| "make rollout dots bigger" | `style.css` | `--rollout-dot-size` |
| "make rollout dots smaller / more space" | `style.css` | `--rollout-dot-gap` |
| "use a different color for high scores" | `style.css` | `--score-3` (and `--score-0..2` for the gradient) |
| "different speaker colors" | `style.css` | `--speaker-1` through `--speaker-8` |
| "dark mode by default" | `style.css` | Move dark vars into `:root` |
| "wider side panel" | `style.css` | `--side-panel-width` |
| "narrower side panel" | `style.css` | `--side-panel-width` |
| "narrower turn nav" | `style.css` | `--turn-nav-width` |
| "smaller / larger fonts" | `style.css` | `--fs-body`, `--fs-small`, `--fs-large` |
| "sans-serif everywhere" | `style.css` | Swap `--font-mono` and `--font-sans` in the body rule |
| "more / less padding" | `style.css` | `--page-padding`, `--gap`, `--gap-loose`, `--gap-section` |
| "change how a conversation looks in the detail view" | `renderers.js` | `renderConversationList()` and the shared `_turnTimelineHtml()` |
| "change how turn-formatted text is parsed" | `renderers.js` | `TURN_RE` regex + `renderTurnText()` |
| "change how logtree HTML is rendered" | `renderers.js` | `renderLogtreeAst()` |
| "change prompt/completion layout to top-bottom instead of side-by-side" | `style.css` | `.pc-grid { grid-template-columns: 1fr; }` |
| "show tool calls differently" | `renderers.js` | `renderEventStream()` — the inner branches per event type |
| "more rollouts per page" | `index.html` | `PAGE_SIZE` constant in the bootstrap script |
| "auto-refresh faster / slower / off" | `index.html` | `setInterval(loadAndRender, 5000)` — change 5000 or remove |
| "collapse metrics by default" | `index.html` | Add `collapsed` to `id="metrics-section"` |
| "expand config by default" | `index.html` | Remove `collapsed` from `id="config-section"` |
| "hide config entirely" | `index.html` | Remove the `config-section` div |
| "different metric chart type" | `renderers.js` | `updateChart()` — currently a line chart, swap the Chart.js `type` field |
| "log scale on y-axis" | `renderers.js` | `updateChart()` options.scales.y → add `type: 'logarithmic'` |
| "no auto-open browser" | `dashboard.py` | Remove the `webbrowser.open()` call in `main()` |
| "different port" | `dashboard.py` | `PORT_BASE` constant at the top |
| "more relaxed step-log parsing" | `dashboard.py` | `_STEP_LOG_RE` regex |

## What to do for changes not in the map

If a user asks for something not in this map, **first check if it's actually one of these in disguise** ("can the dots be more colorful" = `--score-*` vars). Only if it's genuinely something new should you look at the broader code.

For new rollout formats: you need to update three things in lockstep:
1. The skill (`SKILL.md`) so detection writes the new format string into `run.json`.
2. `renderers.js` — add a `render<NewFormat>()` function + add it to the switch in `renderDetail()`.
3. `dashboard.py` — `_build_preview()` and possibly `load_source_records()` if the format needs special server-side prep.

For new aggregate views (e.g., "show me a histogram of rewards"): add a new section to `index.html` and the corresponding render function in `renderers.js`. Don't touch existing renderers.

## What NOT to do

- Don't rewrite `dashboard.py`. It's stable code; the user came here to read rollouts, not to debug the server.
- Don't switch UI frameworks. The whole point of this design is no build step, no dependencies.
- Don't merge two formats' renderers into one "smarter" function. Keep one function per format — it's the surgical-edit contract.
- Don't add framework-specific vocabulary (`if reward in record` etc.). The skill's detection writes the field names into `run.json`; the renderer reads from those names.

#!/usr/bin/env python3
"""Visual display of dozenwazn candidates and validation results.

Two output modes:
  1. Rich terminal display (stats, tables, tier summary)
  2. Interactive HTML canvas visualization with fisheye magnification

Color tier system (Blue + Yellow primary pair):
  Tier 0 — JAIS + o200k single-token (bare):  GREEN (#22c55e)
  Tier 1 — JAIS single-token only:             BLUE (#3b82f6)
  Tier 2 — o200k single-token only (~46):      AMBER (#eab308)
  Tier 3 — Valid word but NOT single-token:     TEAL (#14b8a6) [future]
  Tier 4 — Invalid/non-word:                    WARM GREY (#78716c) [future]
  Tier 5 — Unverified:                          ZINC (#52525b)
"""

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text


console = Console()


# ── Tier classification ──────────────────────────────────────────────

TIER_COLORS = {
    0: '#22c55e',   # Green: both agree
    1: '#3b82f6',   # Blue: JAIS only
    2: '#eab308',   # Amber: o200k only
    3: '#14b8a6',   # Teal: valid, no token
    4: '#78716c',   # Warm grey: invalid
    5: '#52525b',   # Zinc: unverified
}

TIER_NAMES = {
    0: 'JAIS + o200k',
    1: 'JAIS only',
    2: 'o200k only',
    3: 'Valid (no token)',
    4: 'Invalid',
    5: 'Unverified',
}

TIER_RICH_STYLES = {
    0: 'bold green',
    1: 'bold blue',
    2: 'bold yellow',
    3: 'bold cyan',
    4: 'dim rgb(180,170,150)',
    5: 'dim white',
}


def classify_token_tier(record: dict, validity: dict | None = None) -> int:
    """Classify a tokenizer analysis record into a tier (0-5).

    Single-token status wins the pixel (tiers 0–2) regardless of validity —
    validity is shown in the tooltip/modal. Otherwise validation/gold data
    decides tier 3 (real word, no single token) vs 4 (not a real word),
    falling back to 5 (unverified).
    """
    bare = record.get('tokens_bare', {})
    jais_single = bare.get('jais', 99) == 1
    o200k_single = bare.get('o200k_base', 99) == 1

    if jais_single and o200k_single:
        return 0
    elif jais_single:
        return 1
    elif o200k_single:
        return 2
    if validity is not None:
        if validity.get('status') == 'real':
            return 3
        if validity.get('status') == 'not_real':
            return 4
    return 5


def load_validity(data_dir: Path) -> dict:
    """(root, pattern) → validity for tiering and modal details.

    Combines gold.csv and any populated model CSVs through the analysis
    consensus (gold > LLM majority > single LLM). Returns {} when neither
    exists, which keeps the display identical to the pre-validation era.
    """
    from src.analysis import consensus, load_gold, load_model_votes

    gold = load_gold(data_dir) or {}
    votes_by_model = load_model_votes(Path(data_dir))
    keys = set(gold)
    for votes in votes_by_model.values():
        keys.update(votes)

    out = {}
    for key in keys:
        votes = {m: mv[key]['exists'] for m, mv in votes_by_model.items() if key in mv}
        g = gold.get(key)
        cons = consensus(votes, g['real'] if g else None)

        gloss = (g or {}).get('gloss') or ''
        details = None
        for mv in votes_by_model.values():
            v = mv.get(key)
            if v is None:
                continue
            if not gloss:
                gloss = v.get('msa_meanings') or v.get('dialect_meanings') or ''
            if details is None:
                details = v
        out[key] = {
            'status': cons['status'],
            'provenance': cons['provenance'],
            'gloss': gloss,
            'gold': (g or {}).get('gold', ''),
            'details': details,
        }
    return out


# ── Data loading ─────────────────────────────────────────────────────

def load_candidates(path: Path) -> list[dict]:
    """Load candidates from JSONL."""
    with open(path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]


def load_tokenizer_data(path: Path) -> list[dict]:
    """Load tokenizer analysis from JSONL, sorted alphabetically by bare verb."""
    if not path.exists():
        return []
    with open(path, 'r', encoding='utf-8') as f:
        records = [json.loads(line) for line in f]
    records.sort(key=lambda r: r.get('verb_bare', r.get('verb', '')))
    return records


# ── Rich terminal display ────────────────────────────────────────────

def display_stats(candidates: list[dict]):
    total = len(candidates)
    by_arity = Counter(c['arity'] for c in candidates)
    by_type = Counter(c['root_type'] for c in candidates)
    by_pattern = Counter(c['pattern'] for c in candidates)
    unique_roots = len(set(c['root'] for c in candidates))

    table = Table(title="Generation Statistics", show_header=False, border_style="bright_blue")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="white", justify="right")
    table.add_row("Total candidates", f"{total:,}")
    table.add_row("Unique roots", f"{unique_roots:,}")
    table.add_row("", "")
    table.add_row("[bold]By arity[/bold]", "")
    table.add_row("  Trilateral (3)", f"{by_arity.get(3, 0):,}")
    table.add_row("  Quadrilateral (4)", f"{by_arity.get(4, 0):,}")
    table.add_row("", "")
    table.add_row("[bold]By root type[/bold]", "")
    for rt in ['XYZ', 'XYY', 'XYX', 'XYXY', 'XYZN']:
        if rt in by_type:
            table.add_row(f"  {rt}", f"{by_type[rt]:,}")
    table.add_row("", "")
    table.add_row("[bold]By pattern[/bold]", "")
    for pat, count in by_pattern.most_common():
        table.add_row(f"  {pat}", f"{count:,}")
    console.print(table)


def display_tier_summary(records: list[dict], validity_map: dict | None = None):
    validity_map = validity_map or {}
    tier_counts = Counter(
        classify_token_tier(r, validity_map.get((r.get('root', ''), r.get('pattern', ''))))
        for r in records
    )
    total = len(records)
    table = Table(title="Token Tier Distribution", border_style="bright_blue")
    table.add_column("Tier", style="bold")
    table.add_column("Color")
    table.add_column("Count", justify="right")
    table.add_column("Percent", justify="right")
    for tier in range(6):
        c = tier_counts.get(tier, 0)
        if c == 0:
            continue
        style = TIER_RICH_STYLES[tier]
        table.add_row(
            Text(TIER_NAMES[tier], style=style),
            Text("\u25cf", style=style),
            f"{c:,}", f"{100*c/total:.1f}%",
        )
    console.print(table)


# ── Interactive HTML visualization ───────────────────────────────────

def generate_html(records: list[dict], output_path: Path,
                  validity_map: dict | None = None,
                  neologisms: list | None = None):
    """Generate the interactive canvas-based HTML visualization.

    validity_map (from load_validity) drives tiers 3/4, the modal's
    validity row, and the embedded VD details array (real/dialect verbs
    only, to keep the self-contained HTML small). neologisms is a list of
    (root, pattern) keys to expose as a highlight toggle.
    """
    validity_map = validity_map or {}
    neo_keys = set(neologisms or [])

    # Classify tiers and build compact data
    tier_counts = Counter()
    compact_data = []
    details_arr = []   # VD: [gloss, dialects, msa_meanings, dialect_meanings]
    neo_indices = []
    for idx, r in enumerate(records):
        key = (r.get('root', ''), r.get('pattern', ''))
        validity = validity_map.get(key)
        tier = classify_token_tier(r, validity)
        tier_counts[tier] += 1
        bare = r.get('tokens_bare', {})
        cell = {
            'v': r.get('verb', ''),
            'b': r.get('verb_bare', ''),
            'r': r.get('root', ''),
            'rt': r.get('root_type', ''),
            'a': r.get('arity', 3),
            'p': r.get('pattern', ''),
            'tj': bare.get('jais', -1),
            'to': bare.get('o200k_base', -1),
            'tc': bare.get('cl100k_base', -1),
            'w': r.get('weak_class') or '',
            't': tier,
        }
        if validity is not None and validity['status'] != 'unverified':
            cell['val'] = 1 if validity['status'] == 'real' else 2
            cell['vp'] = {'gold': 'g', 'llm-majority': 'm', 'llm-single': 's'}.get(
                validity['provenance'], '')
            d = validity.get('details') or {}
            # Embed details only where there is something to show, and only
            # for real/dialect verbs — keeps the payload small.
            if validity['status'] == 'real' and (validity['gloss'] or d):
                cell['di'] = len(details_arr)
                details_arr.append([
                    validity.get('gloss', ''),
                    '|'.join(d.get('dialects', [])),
                    d.get('msa_meanings', ''),
                    d.get('dialect_meanings', ''),
                ])
        if key in neo_keys:
            neo_indices.append(idx)
        compact_data.append(cell)

    total = len(compact_data)
    data_json = json.dumps(compact_data, ensure_ascii=False, separators=(',', ':'))
    details_json = json.dumps(details_arr, ensure_ascii=False, separators=(',', ':'))
    neo_json = json.dumps(neo_indices, separators=(',', ':'))

    html = f"""<!DOCTYPE html>
<html lang="ar">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Dozenwazn \u2014 \u0623\u0648\u0632\u0627\u0646</title>
<link href="https://fonts.googleapis.com/css2?family=Noto+Sans+Arabic:wght@400;600;700&family=Inter:wght@400;600;700&display=swap" rel="stylesheet">
<style>
:root {{
  /* Company palette */
  --color-charcoal: #1B1A16;
  --color-red: #DA2E22;
  --color-orange: #F59F38;
  --color-cream: #F0ECDB;
  --color-warm-white: #fff7ea;

  /* App theme */
  --bg: var(--color-cream);
  --surface: var(--color-warm-white);
  --surface-hover: #faf5e8;
  --border: #D4CCB8;
  --text: var(--color-charcoal);
  --text-dim: #8B7A72;

  /* Tier colors — complementary to warm red/orange palette */
  --tier0: #1A7A5A;   /* Deep emerald green: JAIS+o200k (strongest) */
  --tier1: #2B6A99;   /* Steel blue: JAIS only */
  --tier2: #F59F38;   /* Company orange: o200k only */
  --tier3: #1A7A5A;   /* Emerald: valid, no token */
  --tier4: #B88A7E;   /* Muted red-brown: invalid */
  --tier5: #C9C0AD;   /* Light charcoal wash: unverified */
}}
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
html, body {{ height: 100%; overflow: hidden; }}
body {{
  font-family: 'Inter', system-ui, sans-serif;
  background: var(--bg);
  color: var(--text);
  display: flex;
  flex-direction: column;
}}

/* ── Header ── */
header {{
  padding: 1rem 2rem 0.8rem;
  flex-shrink: 0;
  background: linear-gradient(180deg, var(--color-warm-white), var(--bg));
  border-bottom: 2px solid var(--border);
}}
h1 {{
  font-size: 2rem;
  text-align: center;
  background: linear-gradient(135deg, var(--color-red), var(--color-orange));
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  font-weight: 800;
  margin-bottom: 0.2rem;
}}
.subtitle {{
  text-align: center;
  color: var(--text-dim);
  font-size: 0.95rem;
  margin-bottom: 0.8rem;
}}
.stats {{
  display: flex;
  gap: 0.6rem;
  justify-content: center;
  flex-wrap: wrap;
  margin-bottom: 0.6rem;
}}
.stat {{
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 0.5rem 1rem;
  text-align: center;
  min-width: 110px;
  box-shadow: 0 1px 3px rgba(27,26,22,0.08);
}}
.stat .n {{ font-size: 1.3rem; font-weight: 700; }}
.stat .l {{ font-size: 0.7rem; color: var(--text-dim); }}
.stat.t0 .n {{ color: var(--tier0); }}
.stat.t1 .n {{ color: var(--tier1); }}
.stat.t2 .n {{ color: var(--tier2); }}
.stat.t5 .n {{ color: var(--color-charcoal); opacity: 0.4; }}

.legend {{
  display: flex;
  gap: 1.2rem;
  justify-content: center;
  flex-wrap: wrap;
  font-size: 0.78rem;
  color: var(--text-dim);
}}
.legend-item {{
  display: flex; align-items: center; gap: 0.3rem;
  background: none; border: 1px solid transparent; border-radius: 6px;
  padding: 0.15rem 0.45rem; cursor: pointer;
  font: inherit; color: inherit;
}}
.legend-item:hover {{ border-color: rgba(27,26,22,0.25); }}
.legend-item.off {{ opacity: 0.35; }}
.legend-item.on {{ border-color: rgba(27,26,22,0.45); background: rgba(27,26,22,0.06); }}
.legend-dot {{ width: 10px; height: 10px; border-radius: 50%; flex-shrink: 0; }}

/* ── Grid container ── */
#grid-wrap {{
  flex: 1;
  overflow: auto;
  position: relative;
  cursor: crosshair;
}}
canvas {{
  display: block;
}}

/* ── Tooltip ── */
#tooltip {{
  position: fixed;
  pointer-events: none;
  z-index: 100;
  background: var(--color-warm-white);
  border: 1px solid var(--border);
  border-radius: 8px;
  box-shadow: 0 4px 16px rgba(27,26,22,0.15);
  padding: 0.6rem 1rem;
  max-width: 300px;
  font-size: 0.85rem;
  opacity: 0;
  transition: opacity 0.15s;
  backdrop-filter: blur(8px);
  direction: rtl;
}}
#tooltip.show {{ opacity: 1; }}
#tooltip .tt-verb {{
  font-family: 'Noto Sans Arabic', serif;
  font-size: 1.4rem;
  font-weight: 700;
  margin-bottom: 0.3rem;
}}
#tooltip .tt-info {{ color: var(--text-dim); font-size: 0.8rem; direction: ltr; }}

/* ── Modal ── */
#backdrop {{
  position: fixed; inset: 0;
  background: rgba(27,26,22,0.5);
  z-index: 200;
  display: none;
  align-items: center;
  justify-content: center;
  backdrop-filter: blur(4px);
}}
#backdrop.open {{ display: flex; }}
#modal {{
  background: var(--color-warm-white);
  border: 1px solid var(--border);
  border-radius: 16px;
  padding: 2rem;
  max-width: 500px;
  width: 90%;
  position: relative;
  direction: rtl;
  box-shadow: 0 8px 32px rgba(27,26,22,0.2);
}}
#modal .close {{
  position: absolute;
  top: 0.8rem;
  left: 0.8rem;
  background: none;
  border: none;
  color: var(--text-dim);
  font-size: 1.5rem;
  cursor: pointer;
  line-height: 1;
}}
#modal .close:hover {{ color: var(--text); }}
#modal .m-verb {{
  font-family: 'Noto Sans Arabic', serif;
  font-size: 3rem;
  font-weight: 700;
  text-align: center;
  margin-bottom: 1rem;
}}
#modal .m-grid {{
  display: grid;
  grid-template-columns: auto 1fr;
  gap: 0.3rem 1rem;
  font-size: 0.9rem;
  direction: ltr;
}}
#modal .m-label {{ color: var(--text-dim); text-align: right; }}
#modal .m-val {{ font-weight: 600; }}
#modal .m-method {{
  margin-top: 1.2rem;
  padding-top: 1rem;
  border-top: 1px solid var(--border);
  font-size: 0.82rem;
  color: var(--text-dim);
  line-height: 1.5;
  direction: ltr;
}}
</style>
</head>
<body>
<header>
  <h1>Dozenwazn</h1>
  <p class="subtitle">All {total:,} mathematically possible Arabic Form I verbs \u00b7 hover to magnify \u00b7 click to explore</p>
  <div class="stats">
    <div class="stat"><div class="n">{total:,}</div><div class="l">Total</div></div>
    <div class="stat t0"><div class="n">{tier_counts.get(0,0):,}</div><div class="l">JAIS + o200k</div></div>
    <div class="stat t1"><div class="n">{tier_counts.get(1,0):,}</div><div class="l">JAIS only</div></div>
    <div class="stat t2"><div class="n">{tier_counts.get(2,0):,}</div><div class="l">o200k only</div></div>
    <div class="stat t3"><div class="n">{tier_counts.get(3,0):,}</div><div class="l">Valid (no token)</div></div>
    <div class="stat t4"><div class="n">{tier_counts.get(4,0):,}</div><div class="l">Invalid</div></div>
    <div class="stat t5"><div class="n">{tier_counts.get(5,0):,}</div><div class="l">Unverified</div></div>
  </div>
  <div class="legend">
    <button class="legend-item" data-tier="0"><div class="legend-dot" style="background:var(--tier0)"></div>JAIS + o200k (strongest)</button>
    <button class="legend-item" data-tier="1"><div class="legend-dot" style="background:var(--tier1)"></div>JAIS only</button>
    <button class="legend-item" data-tier="2"><div class="legend-dot" style="background:var(--tier2)"></div>o200k only (investigate)</button>
    <button class="legend-item" data-tier="3"><div class="legend-dot" style="background:var(--tier3)"></div>Valid (no token)</button>
    <button class="legend-item" data-tier="4"><div class="legend-dot" style="background:var(--tier4)"></div>Invalid</button>
    <button class="legend-item" data-tier="5"><div class="legend-dot" style="background:var(--tier5)"></div>Unverified</button>
    <button class="legend-item" id="neo-toggle" hidden>⭐ Top neologisms</button>
  </div>
</header>

<div id="grid-wrap">
  <canvas id="grid"></canvas>
</div>

<div id="tooltip">
  <div class="tt-verb"></div>
  <div class="tt-info"></div>
</div>

<div id="backdrop">
  <div id="modal">
    <button class="close">\u00d7</button>
    <div class="m-verb"></div>
    <div class="m-grid"></div>
    <div class="m-method"></div>
  </div>
</div>

<script>
// ── Data ──
const D = {data_json};
const N = D.length;
const VD = {details_json};   // [gloss, dialects, msa_meanings, dialect_meanings]
const NEO = {neo_json};      // indices of top neologism candidates
const VAL_NAMES = ['', 'Real', 'Not real'];
const PROV_NAMES = {{g: 'dictionary', m: 'LLM majority', s: 'single LLM'}};

// ── Tier colors ──
const TC = ['#1A7A5A','#2B6A99','#F59F38','#14b8a6','#B88A7E','#C9C0AD'];
const TN = ['JAIS + o200k','JAIS only','o200k only','Valid (no token)','Invalid','Unverified'];

// ── Tier filters ──
const DIM = '#E7E2D2';
const activeTiers = new Set([0, 1, 2, 3, 4, 5]);
const neoSet = new Set(NEO);
let neoMode = false;

function cellVisible(i) {{
  if (neoMode) return neoSet.has(i);
  return activeTiers.has(D[i].t);
}}

// ── Grid config ──
const COLS = 180;
const CELL_W = 32;
const CELL_H = 20;
const BASE_FONT = 11;
const ROWS = Math.ceil(N / COLS);
const GRID_W = COLS * CELL_W;
const GRID_H = ROWS * CELL_H;

// ── Canvas setup ──
const canvas = document.getElementById('grid');
const ctx = canvas.getContext('2d');
const wrap = document.getElementById('grid-wrap');
const tooltip = document.getElementById('tooltip');
const backdrop = document.getElementById('backdrop');
const modal = document.getElementById('modal');

canvas.width = GRID_W;
canvas.height = GRID_H;
canvas.style.width = GRID_W + 'px';
canvas.style.height = GRID_H + 'px';

// ── Spatial index ──
function idxAt(px, py) {{
  const col = Math.floor(px / CELL_W);
  const row = Math.floor(py / CELL_H);
  if (col < 0 || col >= COLS || row < 0 || row >= ROWS) return -1;
  const i = row * COLS + col;
  return i < N ? i : -1;
}}

function cellXY(i) {{
  const col = i % COLS;
  const row = Math.floor(i / COLS);
  return [col * CELL_W, row * CELL_H];
}}

// ── Drawing ──
let mouseX = -1, mouseY = -1;
let hovIdx = -1;
let dirty = true;

const FISHEYE_R = 15;
const MAX_SCALE = 6;
const SIGMA = 1.4;  // Gaussian width — smaller = tighter focus

function tierColor(t) {{ return TC[t] || TC[5]; }}

function drawBase() {{
  ctx.clearRect(0, 0, GRID_W, GRID_H);
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.font = BASE_FONT + 'px "Noto Sans Arabic"';

  // Visible range
  const st = wrap.scrollTop;
  const sl = wrap.scrollLeft;
  const vw = wrap.clientWidth;
  const vh = wrap.clientHeight;

  const r0 = Math.max(0, Math.floor(st / CELL_H) - 1);
  const r1 = Math.min(ROWS, Math.ceil((st + vh) / CELL_H) + 1);
  const c0 = Math.max(0, Math.floor(sl / CELL_W) - 1);
  const c1 = Math.min(COLS, Math.ceil((sl + vw) / CELL_W) + 1);

  for (let row = r0; row < r1; row++) {{
    for (let col = c0; col < c1; col++) {{
      const i = row * COLS + col;
      if (i >= N) break;
      const d = D[i];
      // Dim, don't hide: cells keep their positions so idxAt/tooltip/modal
      // stay valid with zero layout changes.
      ctx.fillStyle = cellVisible(i) ? tierColor(d.t) : DIM;
      const x = col * CELL_W + CELL_W / 2;
      const y = row * CELL_H + CELL_H / 2;
      ctx.fillText(d.b, x, y);
    }}
  }}
}}

function drawFisheye() {{
  if (mouseX < 0) return;

  const focalCol = mouseX / CELL_W;
  const focalRow = mouseY / CELL_H;

  // Extra margin beyond FISHEYE_R to account for displacement of inner cells
  const MARGIN = 3;
  const r0 = Math.max(0, Math.floor(focalRow) - FISHEYE_R - MARGIN);
  const r1 = Math.min(ROWS, Math.ceil(focalRow) + FISHEYE_R + MARGIN + 1);
  const c0 = Math.max(0, Math.floor(focalCol) - FISHEYE_R - MARGIN);
  const c1 = Math.min(COLS, Math.ceil(focalCol) + FISHEYE_R + MARGIN + 1);

  // Clear just the region we'll redraw, plus small pixel pad for glow/shadow
  const pixPad = 20;
  const clearX = c0 * CELL_W - pixPad;
  const clearY = r0 * CELL_H - pixPad;
  const clearW = (c1 - c0) * CELL_W + pixPad * 2;
  const clearH = (r1 - r0) * CELL_H + pixPad * 2;
  ctx.clearRect(clearX, clearY, clearW, clearH);

  // Draw with Gaussian fisheye displacement
  for (let row = r0; row < r1; row++) {{
    for (let col = c0; col < c1; col++) {{
      const i = row * COLS + col;
      if (i >= N) continue;
      const d = D[i];

      const dx = col + 0.5 - focalCol;
      const dy = row + 0.5 - focalRow;
      const dist = Math.sqrt(dx * dx + dy * dy);

      let scale = 1;
      if (dist < FISHEYE_R) {{
        scale = 1 + (MAX_SCALE - 1) * Math.exp(-dist * dist / (2 * SIGMA * SIGMA));
      }}

      const fontSize = Math.round(BASE_FONT * scale);
      ctx.font = fontSize + 'px "Noto Sans Arabic"';

      // Displace outward from focal point — push neighbors away proportional to scale
      const cx = col * CELL_W + CELL_W / 2;
      const cy = row * CELL_H + CELL_H / 2;
      const ox = dx * (scale - 1) * CELL_W * 0.65;
      const oy = dy * (scale - 1) * CELL_H * 0.65;

      const visible = cellVisible(i);
      // Glow for large scales
      if (scale > 2 && visible) {{
        ctx.shadowColor = tierColor(d.t);
        ctx.shadowBlur = (scale - 1) * 3;
      }} else {{
        ctx.shadowColor = 'transparent';
        ctx.shadowBlur = 0;
      }}

      ctx.fillStyle = visible ? tierColor(d.t) : DIM;
      if (scale > 1.8) {{
        // Show diacriticized form when magnified enough
        ctx.fillText(d.v, cx + ox, cy + oy);
      }} else {{
        ctx.fillText(d.b, cx + ox, cy + oy);
      }}
    }}
  }}

  // Reset shadow
  ctx.shadowColor = 'transparent';
  ctx.shadowBlur = 0;
  ctx.font = BASE_FONT + 'px "Noto Sans Arabic"';
}}

function draw() {{
  if (!dirty) {{ requestAnimationFrame(draw); return; }}
  dirty = false;
  drawBase();
  drawFisheye();
  requestAnimationFrame(draw);
}}

// ── Interactions ──
let hoverTimer = null;
let lastHovIdx = -1;

function showTooltip(i, px, py) {{
  const d = D[i];
  const ttVerb = tooltip.querySelector('.tt-verb');
  const ttInfo = tooltip.querySelector('.tt-info');

  ttVerb.textContent = d.v;
  ttVerb.style.color = tierColor(d.t);

  const validityNote = d.val ? ' \\u00b7 ' + VAL_NAMES[d.val] +
    (d.vp ? ' (' + PROV_NAMES[d.vp] + ')' : '') : '';
  if (d.t === 4) {{
    ttInfo.textContent = 'Not a real word' + (d.vp ? ' \\u2014 per ' + PROV_NAMES[d.vp] : '') +
      ' \\u00b7 ' + d.r + ' \\u00b7 ' + d.p;
  }} else if (d.t === 5) {{
    ttInfo.textContent = 'Unverified \\u2014 no dictionary or LLM verdict yet';
  }} else if (d.t <= 2) {{
    ttInfo.textContent = d.r + ' (' + d.rt + ') \\u00b7 ' + TN[d.t] + ' \\u00b7 ' + d.p + (d.w ? ' \\u00b7 ' + d.w : '') + validityNote;
  }} else {{
    ttInfo.textContent = d.r + ' (' + d.rt + ') \\u00b7 ' + d.p + (d.w ? ' \\u00b7 ' + d.w : '') + validityNote;
  }}

  // Position tooltip
  const pad = 16;
  let tx = px + pad;
  let ty = py - 60;
  if (tx + 300 > window.innerWidth) tx = px - 300 - pad;
  if (ty < 10) ty = py + pad;
  tooltip.style.left = tx + 'px';
  tooltip.style.top = ty + 'px';
  tooltip.classList.add('show');
}}

function hideTooltip() {{
  tooltip.classList.remove('show');
  clearTimeout(hoverTimer);
  hoverTimer = null;
}}

wrap.addEventListener('mousemove', (e) => {{
  const rect = canvas.getBoundingClientRect();
  const sx = wrap.scrollLeft;
  const sy = wrap.scrollTop;
  mouseX = e.clientX - rect.left + sx;
  mouseY = e.clientY - rect.top + sy;
  dirty = true;

  const i = idxAt(mouseX, mouseY);
  hovIdx = i;

  if (i !== lastHovIdx) {{
    hideTooltip();
    lastHovIdx = i;
    if (i >= 0) {{
      hoverTimer = setTimeout(() => showTooltip(i, e.clientX, e.clientY), 1500);
    }}
  }}
}});

wrap.addEventListener('mouseleave', () => {{
  mouseX = -1;
  mouseY = -1;
  hovIdx = -1;
  lastHovIdx = -1;
  dirty = true;
  hideTooltip();
}});

wrap.addEventListener('scroll', () => {{ dirty = true; }});

// ── Click: detail modal ──
wrap.addEventListener('click', (e) => {{
  const rect = canvas.getBoundingClientRect();
  const sx = wrap.scrollLeft;
  const sy = wrap.scrollTop;
  const px = e.clientX - rect.left + sx;
  const py = e.clientY - rect.top + sy;
  const i = idxAt(px, py);
  if (i < 0) return;

  const d = D[i];
  modal.querySelector('.m-verb').textContent = d.v;
  modal.querySelector('.m-verb').style.color = tierColor(d.t);

  const grid = modal.querySelector('.m-grid');
  grid.innerHTML = '';
  const fields = [
    ['Root', d.r],
    ['Root Type', d.rt],
    ['Arity', d.a],
    ['Pattern', d.p],
    ['Weak Class', d.w || 'n/a'],
    ['Bare Form', d.b],
    ['Tier', TN[d.t]],
    ['Validity', d.val ? VAL_NAMES[d.val] + (d.vp ? ' (' + PROV_NAMES[d.vp] + ')' : '') : 'Unverified'],
    ['JAIS tokens (bare)', d.tj === -1 ? 'N/A' : d.tj],
    ['o200k tokens (bare)', d.to === -1 ? 'N/A' : d.to],
    ['cl100k tokens (bare)', d.tc === -1 ? 'N/A' : d.tc],
  ];
  if (d.di !== undefined) {{
    const det = VD[d.di];
    if (det[0]) fields.push(['Meaning', det[0]]);
    if (det[1]) fields.push(['Dialects', det[1]]);
    if (det[2] && det[2] !== det[0]) fields.push(['MSA meaning', det[2]]);
    if (det[3]) fields.push(['Dialect meaning', det[3]]);
  }}
  for (const [label, val] of fields) {{
    const lbl = document.createElement('div');
    lbl.className = 'm-label';
    lbl.textContent = label;
    const v = document.createElement('div');
    v.className = 'm-val';
    v.textContent = val;
    if (label === 'Tier') v.style.color = tierColor(d.t);
    grid.appendChild(lbl);
    grid.appendChild(v);
  }}

  const method = modal.querySelector('.m-method');
  const provTail = d.vp === 'g'
    ? 'Verdict source: machine-readable dictionaries (Arramooz / Wiktionary).'
    : d.vp === 'm' ? 'Verdict source: majority of local LLM judges.'
    : d.vp === 's' ? 'Verdict source: a single LLM judge (provisional).' : '';
  if (d.t === 4) {{
    method.innerHTML = '<strong>Methodology:</strong> This candidate was generated by combinatorially applying all 28 Arabic consonants to the Form I templates and checked against dictionaries and LLM judges. It does not correspond to a known Arabic word. ' + provTail;
  }} else if (d.t === 3) {{
    method.innerHTML = '<strong>Methodology:</strong> A real Arabic verb that no tokenizer in the panel encodes as a single token \\u2014 the coverage gap this project measures. ' + provTail;
  }} else if (d.t === 5) {{
    method.innerHTML = '<strong>Methodology:</strong> Generated combinatorially from all valid root patterns. Tokenizer analysis shows <strong>' + d.tj + '</strong> token(s) in JAIS and <strong>' + d.to + '</strong> in o200k. No dictionary or LLM verdict yet \\u2014 validation is pending or was inconclusive.';
  }} else {{
    method.innerHTML = '<strong>Tokenizer signal:</strong> Recognized as a single token in <strong>' + TN[d.t] + '</strong>. Words that tokenizers encode as single tokens are typically frequent and well-established in the training corpus. ' + (d.val ? 'Validity: ' + VAL_NAMES[d.val] + '. ' + provTail : 'Validation pending.');
  }}

  backdrop.classList.add('open');
  hideTooltip();
}});

backdrop.addEventListener('click', (e) => {{
  if (e.target === backdrop) backdrop.classList.remove('open');
}});
modal.querySelector('.close').addEventListener('click', () => backdrop.classList.remove('open'));
document.addEventListener('keydown', (e) => {{
  if (e.key === 'Escape') backdrop.classList.remove('open');
}});

// ── Tier filter buttons ──
document.querySelectorAll('.legend-item[data-tier]').forEach((btn) => {{
  btn.addEventListener('click', (e) => {{
    const t = parseInt(btn.dataset.tier, 10);
    neoMode = false;
    document.getElementById('neo-toggle').classList.remove('on');
    if (e.shiftKey) {{
      // Shift-click solos the tier (click again with shift to restore all)
      if (activeTiers.size === 1 && activeTiers.has(t)) {{
        for (let k = 0; k <= 5; k++) activeTiers.add(k);
      }} else {{
        activeTiers.clear();
        activeTiers.add(t);
      }}
    }} else {{
      if (activeTiers.has(t)) activeTiers.delete(t); else activeTiers.add(t);
    }}
    document.querySelectorAll('.legend-item[data-tier]').forEach((b) => {{
      b.classList.toggle('off', !activeTiers.has(parseInt(b.dataset.tier, 10)));
    }});
    dirty = true;
  }});
}});

const neoToggle = document.getElementById('neo-toggle');
if (NEO.length > 0) {{
  neoToggle.hidden = false;
  neoToggle.addEventListener('click', () => {{
    neoMode = !neoMode;
    neoToggle.classList.toggle('on', neoMode);
    document.querySelectorAll('.legend-item[data-tier]').forEach((b) => {{
      b.classList.toggle('off', neoMode);
    }});
    if (!neoMode) {{
      document.querySelectorAll('.legend-item[data-tier]').forEach((b) => {{
        b.classList.toggle('off', !activeTiers.has(parseInt(b.dataset.tier, 10)));
      }});
    }}
    dirty = true;
  }});
}}

// ── Init ──
document.fonts.ready.then(() => {{
  dirty = true;
  requestAnimationFrame(draw);
}});
</script>
</body>
</html>"""

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)
    return output_path


# ── Main ─────────────────────────────────────────────────────────────

if __name__ == '__main__':
    tok_path = Path('data/tokenizer_analysis.jsonl')
    candidates_path = Path('data/candidates.jsonl')

    if not tok_path.exists() and not candidates_path.exists():
        print("Run src/generate.py and src/tokenizer_check.py first.", file=sys.stderr)
        sys.exit(1)

    # Terminal display
    if candidates_path.exists():
        candidates = load_candidates(candidates_path)
        console.print(Panel("[bold bright_blue]\u0623\u0648\u0632\u0627\u0646 \u2014 Dozenwazn[/bold bright_blue]",
                            subtitle="Arabic Form I Verb Generator"))
        console.print()
        display_stats(candidates)

    if tok_path.exists():
        records = load_tokenizer_data(tok_path)
        console.print()
        display_tier_summary(records)

        # HTML export
        html_path = Path('data/dozenwazn.html')
        generate_html(records, html_path)
        console.print(f"\n[dim]Interactive HTML \u2192 {html_path}[/dim]")
    else:
        console.print("\n[dim]Run tokenizer_check.py to enable HTML export.[/dim]")

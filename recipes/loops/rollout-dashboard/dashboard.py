#!/usr/bin/env python3
"""training-explorer dashboard server.

Reads run.json (in the same directory) and serves the dashboard at localhost.

This is a closed system. It is *not* regenerated per run. If you want a
different look, edit style.css. If you want a different rollout layout,
edit renderers.js. If you want a new format, add to BOTH:
  1. The skill's scoring/identification logic (in SKILL.md)
  2. The render*() function in renderers.js
  3. The format dispatch table in this file (see `load_source_records` below)
"""

from __future__ import annotations

import csv
import io
import json
import mimetypes
import os
import re
import sys
import threading
import time
import urllib.parse
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------

PORT_BASE = 8765
PORT_TRIES = 10

RUN_DIR = Path(__file__).resolve().parent
CONFIG_PATH = RUN_DIR / "run.json"

# Static file extensions we serve from RUN_DIR
STATIC = {
    ".html": "text/html; charset=utf-8",
    ".css":  "text/css; charset=utf-8",
    ".js":   "application/javascript; charset=utf-8",
}

# ----------------------------------------------------------------------------
# Config loading
# ----------------------------------------------------------------------------

def load_config():
    if not CONFIG_PATH.exists():
        return {"run_id": RUN_DIR.name, "rollout_sources": [],
                "_error": f"run.json not found at {CONFIG_PATH}"}
    try:
        return json.loads(CONFIG_PATH.read_text())
    except json.JSONDecodeError as e:
        return {"run_id": RUN_DIR.name, "rollout_sources": [],
                "_error": f"run.json failed to parse: {e}"}


# ----------------------------------------------------------------------------
# Data loading — reads the underlying rollout/metrics/config files
# ----------------------------------------------------------------------------

def resolve_paths(file_or_glob: str):
    """Return list of paths matching a file string or glob pattern."""
    if "*" in file_or_glob:
        return sorted(RUN_DIR.glob(file_or_glob))
    p = RUN_DIR / file_or_glob
    return [p] if p.exists() else []


def read_records_from_path(path: Path):
    """Read records from one file. Returns list[dict]."""
    suffix = path.suffix.lower()
    try:
        if suffix == ".jsonl":
            with path.open() as f:
                out = []
                for line in f:
                    line = line.strip()
                    if not line: continue
                    try:
                        out.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
                return out

        if suffix == ".json":
            data = json.loads(path.read_text())
            if isinstance(data, list):
                return data
            if isinstance(data, dict):
                # Single dict with one array-of-dicts field → use it
                array_keys = [k for k, v in data.items()
                              if isinstance(v, list) and v and isinstance(v[0], dict)]
                if len(array_keys) == 1:
                    return data[array_keys[0]]
                return [data]
            return []

        if suffix == ".csv":
            with path.open() as f:
                rows = list(csv.DictReader(f))
            # Coerce numeric strings
            for r in rows:
                for k, v in list(r.items()):
                    if isinstance(v, str):
                        try:
                            if "." in v or "e" in v.lower():
                                r[k] = float(v)
                            else:
                                r[k] = int(v)
                        except (ValueError, AttributeError):
                            pass
            return rows

        if suffix == ".html":
            # For logtree HTML files: parse to a tag/children/attrs AST so
            # renderers.js can treat it the same as embedded ASTs.
            return [{"root": _html_to_ast(path.read_text(errors="replace")),
                     "_path": str(path),
                     "_filename": path.name}]

        if suffix in (".log", ".txt"):
            # Try step-structured log parsing
            return _parse_step_log(path)

    except Exception as e:
        return [{"_error": f"Failed to read {path}: {e}"}]
    return []


# Minimal HTML-to-AST. Output shape matches what logtree libraries write:
#   {"tag": "p", "attrs": {"class": "lt-p"}, "children": ["text..."]}
_HTML_TAG = re.compile(r'<(/?)([a-zA-Z][\w-]*)([^>]*?)(/?)>')
_HTML_ATTR = re.compile(r'(\w[\w-]*)\s*=\s*(?:"([^"]*)"|\'([^\']*)\'|([^\s>]+))')
_VOID_TAGS = {"br", "hr", "img", "input", "meta", "link", "source"}

def _html_to_ast(s: str):
    # Strip everything before <body> if present
    m = re.search(r"<body[^>]*>", s, re.IGNORECASE)
    if m:
        s = s[m.end():]
        end = re.search(r"</body>", s, re.IGNORECASE)
        if end: s = s[:end.start()]

    pos = 0
    root = {"tag": "body", "attrs": {}, "children": []}
    stack = [root]

    while pos < len(s):
        m = _HTML_TAG.search(s, pos)
        if not m:
            text = s[pos:].strip()
            if text:
                stack[-1]["children"].append(text)
            break
        text = s[pos:m.start()].strip()
        if text:
            stack[-1]["children"].append(text)
        pos = m.end()

        closing, tag, attr_blob, self_close = m.group(1), m.group(2).lower(), m.group(3), m.group(4)
        if closing:
            # Pop until matching tag
            while len(stack) > 1 and stack[-1]["tag"] != tag:
                stack.pop()
            if len(stack) > 1: stack.pop()
            continue

        attrs = {}
        for am in _HTML_ATTR.finditer(attr_blob):
            attrs[am.group(1)] = am.group(2) or am.group(3) or am.group(4) or ""

        node = {"tag": tag, "attrs": attrs, "children": []}
        stack[-1]["children"].append(node)

        if tag not in _VOID_TAGS and not self_close:
            stack.append(node)

    return root


# ----------------------------------------------------------------------------
# Conversation extraction — the key Tinker-style logtree handler
# ----------------------------------------------------------------------------

_ROLE_KEYS    = ("role", "speaker", "from", "author", "name")
_CONTENT_KEYS = ("content", "text", "message", "value", "utterance")


def _looks_like_conversation(arr):
    """True iff arr is [{role-ish key, content-ish key}, ...]"""
    if not arr or not isinstance(arr, list):
        return False
    sample = arr[:5]
    if not all(isinstance(x, dict) for x in sample):
        return False
    has_role = sum(any(k in x for k in _ROLE_KEYS) for x in sample)
    has_content = sum(any(k in x for k in _CONTENT_KEYS) for x in sample)
    return has_role >= len(sample) * 0.8 and has_content >= len(sample) * 0.8


def _first_heading_text(node, _depth=0):
    """Find the first h1-h6 inside the AST subtree and return its text."""
    if _depth > 10: return None
    if isinstance(node, dict):
        tag = node.get("tag", "")
        if tag in ("h1", "h2", "h3", "h4", "h5", "h6"):
            return _first_text_leaf(node.get("children", []))
        for c in node.get("children", []):
            r = _first_heading_text(c, _depth + 1)
            if r: return r
    elif isinstance(node, list):
        for c in node:
            r = _first_heading_text(c, _depth + 1)
            if r: return r
    return None


def extract_conversations_from_ast(node, _depth=0):
    """Walk an AST tree, collect every node that contains a conversation.

    A conversation is identified by SHAPE:
      - dict with `data.type == "conversation"` and `data.messages` is a list of
        role/content dicts (Tinker logtree format), OR
      - dict with `data.messages` shaped like a conversation, OR
      - dict with a direct `messages` field shaped like a conversation.

    Returns list of {messages, title, metadata} in document order.
    """
    if _depth > 50: return []
    out = []
    if isinstance(node, dict):
        data = node.get("data")
        if isinstance(data, dict):
            msgs = data.get("messages")
            if isinstance(msgs, list) and _looks_like_conversation(msgs):
                title = data.get("title") or data.get("name") \
                        or _first_heading_text(node) or None
                metadata = {k: v for k, v in data.items()
                            if k not in ("messages", "type")
                            and not isinstance(v, (list, dict))}
                out.append({"messages": msgs, "title": title, "metadata": metadata})
                return out

        msgs = node.get("messages")
        if isinstance(msgs, list) and _looks_like_conversation(msgs):
            out.append({"messages": msgs,
                        "title": node.get("title") or _first_heading_text(node),
                        "metadata": {}})
            return out

        for child in node.get("children", []):
            out.extend(extract_conversations_from_ast(child, _depth + 1))

    elif isinstance(node, list):
        for child in node:
            out.extend(extract_conversations_from_ast(child, _depth + 1))

    return out


def count_conversations_in_file(path: Path):
    """Single-read helper: count conversations in a logtree file. 0 if unparsable."""
    try:
        recs = read_records_from_path(path)
    except Exception:
        return 0
    total = 0
    for r in recs:
        root = r.get("root")
        if root:
            total += len(extract_conversations_from_ast(root))
    return total


# ----------------------------------------------------------------------------
# lt-* HTML AST trajectory splitter (logtree HTML library format)
# ----------------------------------------------------------------------------

_LT_TURN_RE      = re.compile(r"Turn\s+\d+\s*[-\u2013]\s*([^:]+?):\s*(.+)", re.DOTALL)
_LT_GAME_OVER_RE = re.compile(r"Won:\s*([\u2713\u2717])")


_LT_TURN_NUM_RE = re.compile(r"Turn\s+(\d+)\s*[-\u2013]")
_LT_SECRET_RE   = re.compile(r"Secret:\s*([\w\s]+?)(?:,|$)")


def _extract_lt_trajectories(root):
    """Split a logtree HTML AST (lt-* CSS class format) into per-group records.

    Each lt-section (group) becomes one record with all its interleaved turns.
    Turns from parallel trajectories are tagged with their turn number so the
    client renderer can group them as "Turn N: all Player questions, all Answerer
    responses".

    Returns a list of dicts, each with:
      messages      : [{role, content, turn?}, ...]
      group_idx     : int (0-based group number)
      traj_idx      : 0  (placeholder — this record represents the whole group)
      won           : float  (fraction of trajectories that won, 0.0–1.0)
      secret        : str or None
      n_trajectories: int
    """
    records = []

    def nc(n):
        return n.get("attrs", {}).get("class", "") if isinstance(n, dict) else ""

    def all_text(n):
        if isinstance(n, str):   return n
        if isinstance(n, dict):  return " ".join(filter(None, (all_text(c) for c in n.get("children", []))))
        if isinstance(n, list):  return " ".join(filter(None, (all_text(c) for c in n)))
        return ""

    def process_trajectory(traj_node, group_idx):
        """Original per-trajectory path (used when lt-h4 sub-sections exist)."""
        children = traj_node.get("children", [])
        h4 = next((c for c in children if nc(c) == "lt-h4"), None)
        traj_idx = len(records)
        if h4:
            heading = all_text(h4)
            m = re.search(r"Trajectory\s+(\d+)", heading)
            if m: traj_idx = int(m.group(1))
        body = next((c for c in children if nc(c) == "lt-section-body"), None)
        if not body: return
        messages, won = [], None
        for child in body.get("children", []):
            cls = nc(child)
            if cls == "lt-section":
                inner = child.get("children", [])
                if any(nc(c) == "lt-h5" for c in inner):
                    continue
                for sub in child.get("children", []):
                    if nc(sub) == "lt-section-body":
                        for item in sub.get("children", []):
                            if nc(item) == "lt-p":
                                text = all_text(item).strip()
                                m2 = _LT_TURN_RE.match(text)
                                if m2:
                                    messages.append({"role": m2.group(1).strip(),
                                                     "content": m2.group(2).strip()})
            elif cls == "lt-p":
                text = all_text(child).strip()
                go_m = _LT_GAME_OVER_RE.search(text)
                if go_m:
                    won = 1.0 if go_m.group(1) == "\u2713" else 0.0
                    messages.append({"role": "Game Over", "content": text})
        if messages:
            records.append({"messages": messages, "group_idx": group_idx,
                            "traj_idx": traj_idx, "won": won})

    def process_group_flat(body_node, group_idx):
        """Flat path: body contains interleaved lt-p turns from all trajectories.

        Produces ONE group record with all raw interleaved messages.
        Trajectory reconstruction happens client-side in renderGroupTurns.
        """
        msgs = []
        game_over_list = []

        for child in body_node.get("children", []):
            if nc(child) != "lt-p":
                continue
            text = all_text(child).strip()
            if not text:
                continue
            go_m = _LT_GAME_OVER_RE.search(text)
            if go_m:
                won_val = 1.0 if go_m.group(1) == "\u2713" else 0.0
                game_over_list.append({"role": "Game Over", "content": text, "won": won_val})
                continue
            t_m = _LT_TURN_RE.match(text)
            if t_m:
                n_m = _LT_TURN_NUM_RE.match(text)
                turn_num = int(n_m.group(1)) if n_m else None
                msg = {"role": t_m.group(1).strip(), "content": t_m.group(2).strip()}
                if turn_num is not None:
                    msg["turn"] = turn_num
                msgs.append(msg)

        if not msgs and not game_over_list:
            return

        secret = None
        if game_over_list:
            s_m = _LT_SECRET_RE.search(game_over_list[0]["content"])
            if s_m:
                secret = s_m.group(1).strip()

        won_count = sum(1 for go in game_over_list if go["won"] == 1.0)
        n_traj = len(game_over_list)
        won_frac = won_count / n_traj if n_traj > 0 else None

        records.append({
            "messages":      msgs + [{"role": go["role"], "content": go["content"]}
                                     for go in game_over_list],
            "group_idx":     group_idx,
            "won":           won_frac,
            "secret":        secret,
            "n_trajectories": n_traj,
        })

    def process_group(group_node, group_idx):
        for child in group_node.get("children", []):
            if nc(child) == "lt-section-body":
                body_children = child.get("children", [])
                has_traj_sections = any(
                    nc(item) == "lt-section"
                    and any(nc(c) == "lt-h4" for c in item.get("children", []))
                    for item in body_children
                )
                if has_traj_sections:
                    for item in body_children:
                        if nc(item) == "lt-section":
                            inner = item.get("children", [])
                            if any(nc(c) == "lt-h4" for c in inner):
                                process_trajectory(item, group_idx)
                else:
                    process_group_flat(child, group_idx)

    group_idx = 0
    for child in (root.get("children", []) if isinstance(root, dict) else []):
        if nc(child) == "lt-section":
            inner = child.get("children", [])
            if any(nc(c) == "lt-h2" for c in inner):
                process_group(child, group_idx)
                group_idx += 1

    return records


_STEP_LOG_RE = re.compile(
    r"\[?(?:step|iter|iteration|epoch)\s+(\d+)\]?[^=\n]*?"
    r"((?:\w[\w/-]*=[\d.eE+\-]+\s*)+)",
    re.IGNORECASE,
)

def _parse_step_log(path: Path):
    recs = []
    for line in path.read_text(errors="replace").splitlines():
        m = _STEP_LOG_RE.search(line)
        if not m: continue
        rec = {"step": int(m.group(1))}
        for kv in m.group(2).split():
            if "=" in kv:
                k, v = kv.split("=", 1)
                try:
                    rec[k] = float(v) if "." in v else int(v)
                except ValueError:
                    rec[k] = v
        recs.append(rec)
    return recs


def load_source_records(source: dict):
    """Load all records for one rollout source. Returns list[dict]."""
    paths = resolve_paths(source.get("glob") or source["file"])
    all_recs = []

    fmt = source.get("format")
    fields = source.get("fields", {})
    step_from_path = fields.get("step_from_path", False)
    step_field = fields.get("step")
    id_field = fields.get("id")
    is_multifile = len(paths) > 1

    for path_idx, p in enumerate(paths):
        recs = read_records_from_path(p)

        # Logtree splitting: each `root` field that's an AST with conversation
        # nodes becomes N records (one per conversation). This is the key fix
        # for "every parallel trajectory should be its own rollout."
        if fmt == "logtree_ast":
            content_field = fields.get("content", "root")
            split_recs = []
            for r in recs:
                root = r.get(content_field)
                convs = extract_conversations_from_ast(root) if root else []
                if convs:
                    for conv_idx, conv in enumerate(convs):
                        # Each conversation becomes a separate record. Keep parent
                        # metadata (title, started_at, etc.) and replace content
                        # with just this conversation's messages.
                        new_rec = {k: v for k, v in r.items()
                                   if k != content_field and not k.startswith("_")}
                        new_rec[content_field] = conv["messages"]
                        new_rec["_conversation_idx"] = conv_idx
                        new_rec["_conversation_title"] = conv.get("title") or f"Rollout {conv_idx}"
                        new_rec["_source_file"] = str(p.relative_to(RUN_DIR))
                        if "metadata" in conv:
                            for k, v in conv["metadata"].items():
                                # Prefix to avoid clobbering existing keys
                                new_rec.setdefault(f"meta_{k}", v)
                        split_recs.append(new_rec)
                else:
                    # Try lt-* HTML AST format (logtree HTML library)
                    lt_trajs = _extract_lt_trajectories(root) if root else []
                    if lt_trajs:
                        for conv_idx, traj in enumerate(lt_trajs):
                            new_rec = {k: v for k, v in r.items()
                                       if k != content_field and not k.startswith("_")}
                            new_rec[content_field]      = traj["messages"]
                            new_rec["_conversation_idx"] = conv_idx
                            new_rec["group_idx"]         = traj["group_idx"]
                            new_rec["won"]               = traj.get("won")
                            new_rec["secret"]            = traj.get("secret")
                            new_rec["n_trajectories"]    = traj.get("n_trajectories", 0)
                            new_rec["_source_file"]      = str(p.relative_to(RUN_DIR))
                            new_rec["_format_override"]  = "group_turns"
                            split_recs.append(new_rec)
                    else:
                        # No conversations found — keep the original record so we
                        # still render the AST as a tree in the detail view.
                        r["_source_file"] = str(p.relative_to(RUN_DIR))
                        split_recs.append(r)
            recs = split_recs
            # If we split, set format override so the renderer dispatches correctly.
            # group_turns takes priority (already set above); fall back to
            # conversation_list for any remaining list records.
            for r in recs:
                if isinstance(r.get(fields.get("content", "root")), list):
                    r.setdefault("_format_override", "conversation_list")

        if step_from_path and step_field:
            m = re.search(r"(?:iteration|step|epoch)[_-]?(\d+)", str(p), re.IGNORECASE)
            step_val = int(m.group(1)) if m else None
            for r in recs:
                r[step_field] = step_val

        # Synthesize ids — required when no id field, AND when records from
        # multiple files might share natural ids (each file's record-0 collides).
        base = len(all_recs)
        for i, r in enumerate(recs):
            r["_synth_id"] = f"row-{base + i}"
            if "_source_file" not in r:
                r["_source_file"] = str(p.relative_to(RUN_DIR))
            if is_multifile and id_field and id_field in r:
                r["_qualified_id"] = f"{p.relative_to(RUN_DIR)}#{r[id_field]}"
        all_recs.extend(recs)

    # For event streams, group events by rollout_id
    if fmt == "event_stream":
        return _group_events_into_rollouts(all_recs, fields)

    # Attach preview for flat list display
    for r in all_recs:
        r["_preview"] = _build_preview(r, source)

    return all_recs


def _event_type(e):
    """Normalize event type — supports both 'type' and 'event' keys."""
    return e.get("type") or e.get("event") or ""


def _group_events_into_rollouts(events: list, fields: dict):
    """Turn a list of events into a list of rollouts. One rollout per id."""
    id_field = fields.get("id") or "rollout_id"
    # Store events under the content field name so the renderer finds them
    content_field = fields.get("content") or "events"
    by_id = {}
    order = []
    for ev in events:
        rid = ev.get(id_field)
        if rid is None: continue
        if rid not in by_id:
            by_id[rid] = {content_field: [], id_field: rid}
            order.append(rid)
        by_id[rid][content_field].append(ev)

    # Promote useful fields up from events for the list/grid display
    rollouts = []
    for rid in order:
        bucket = by_id[rid]
        evs = bucket[content_field]
        # Promote fields from lifecycle events
        start    = next((e for e in evs if _event_type(e) == "rollout_start"),    None)
        end      = next((e for e in evs if _event_type(e) == "rollout_end"),      None)
        complete = next((e for e in evs if _event_type(e) == "rollout_complete"), None)
        if start:
            for k in ("task", "step"):
                if k in start: bucket[k] = start[k]
        # rollout_end carries reward; rollout_complete carries stop_condition/num_turns
        for ev in (e for e in [end, complete] if e):
            for k in ("reward", "stop_condition", "num_turns", "hit_max_turns"):
                if k in ev and k not in bucket: bucket[k] = ev[k]
        # Preserve in record so renderer can find by id_field
        bucket["_preview"] = _events_preview(evs)
        rollouts.append(bucket)
    return rollouts


def _events_preview(events):
    """First assistant turn or tool call, truncated."""
    for e in events:
        t = _event_type(e)
        if t == "assistant_turn":
            return str(e.get("content") or e.get("preview") or "")[:160]
        if t == "tool_call":
            return f"[{e.get('tool', 'tool')}] " + str(e.get("args") or "")[:140]
    return ""


def _build_preview(record: dict, source: dict):
    """Build a one-line preview for the flat-list row."""
    # Per-record override (set when logtree got split into conversation records)
    fmt = record.get("_format_override") or source.get("format")
    f = source.get("fields", {})
    if fmt == "prompt_completion":
        p = record.get(f.get("content_prompt", ""))
        return str(p or "")[:160]
    if fmt == "conversation_list":
        lst = record.get(f.get("content", ""), [])
        if isinstance(lst, list) and lst:
            # Prefer first user/player turn for the preview
            for item in lst[:3]:
                if isinstance(item, dict):
                    for k in ("content", "text", "message"):
                        if k in item: return str(item[k])[:160]
        return ""
    if fmt in ("turn_text", "raw_text"):
        s = record.get(f.get("content", ""), "")
        return str(s)[:160]
    if fmt == "logtree_ast":
        # Conversation-less logtree — fall back to first text leaf
        return _first_text_leaf(record.get(f.get("content", "")))[:160]
    return ""


def _first_text_leaf(node, _depth=0):
    if _depth > 30: return ""
    if isinstance(node, str): return node
    if isinstance(node, dict): return _first_text_leaf(node.get("children", []), _depth + 1)
    if isinstance(node, list):
        for n in node:
            t = _first_text_leaf(n, _depth + 1)
            if t.strip(): return t
    return ""


def load_metrics_records(metrics_cfg):
    if not metrics_cfg: return []
    paths = resolve_paths(metrics_cfg["file"])
    recs = []
    for p in paths:
        recs.extend(read_records_from_path(p))
    return recs


# ----------------------------------------------------------------------------
# Find a specific rollout for the detail endpoint
# ----------------------------------------------------------------------------

def find_rollout(source, rollout_id):
    """Return the single record matching rollout_id. None if not found."""
    recs = load_source_records(source)
    id_field = (source.get("fields") or {}).get("id")
    rid_str = str(rollout_id)
    # Match against synthetic id first (always unique), then qualified id,
    # then the natural id field.
    for r in recs:
        if str(r.get("_synth_id")) == rid_str: return r
    for r in recs:
        if str(r.get("_qualified_id")) == rid_str: return r
    if id_field:
        for r in recs:
            if str(r.get(id_field)) == rid_str: return r
    return None


# ----------------------------------------------------------------------------
# HTTP handler
# ----------------------------------------------------------------------------

class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):  # silence default access log
        pass

    def _send(self, status, body, content_type="application/json"):
        if isinstance(body, (dict, list)):
            body = json.dumps(body, default=str).encode()
        elif isinstance(body, str):
            body = body.encode()
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        url = urllib.parse.urlparse(self.path)
        path = url.path
        query = urllib.parse.parse_qs(url.query)

        # /data → full RUN payload (config + materialized records)
        if path == "/data":
            cfg = load_config()
            self._materialize_records(cfg)
            return self._send(200, cfg)

        # /detail/<sourceIdx>/<rolloutId>
        if path.startswith("/detail/"):
            parts = path.split("/", 3)
            if len(parts) >= 4:
                try:
                    source_idx = int(parts[2])
                except ValueError:
                    return self._send(400, {"error": "bad source index"})
                rollout_id = urllib.parse.unquote(parts[3])

                cfg = load_config()
                sources = cfg.get("rollout_sources", [])
                if not (0 <= source_idx < len(sources)):
                    return self._send(404, {"error": "source not found"})
                source = sources[source_idx]
                record = find_rollout(source, rollout_id)
                if record is None:
                    return self._send(404, {"error": f"rollout {rollout_id!r} not found"})

                # JSON mode: return the data for the iframe page to render
                if "json" in query:
                    # Apply per-record format override so the renderer dispatches
                    # correctly (e.g. conversation_list for lt-trajectory splits).
                    resp_source = dict(source)
                    if record.get("_format_override"):
                        resp_source["format"] = record["_format_override"]
                    return self._send(200, {"source": resp_source, "record": record})

                # Otherwise serve detail.html
                return self._serve_static("detail.html")

        # Static files (index.html, style.css, renderers.js, detail.html)
        if path == "/":
            return self._serve_static("index.html")
        if path in ("/index.html", "/style.css", "/renderers.js", "/detail.html"):
            return self._serve_static(path.lstrip("/"))

        return self._send(404, {"error": "not found", "path": path})

    def _serve_static(self, name):
        p = RUN_DIR / name
        if not p.exists():
            return self._send(404, {"error": f"missing file: {name}"})
        ct = STATIC.get(p.suffix.lower(), "application/octet-stream")
        return self._send(200, p.read_bytes(), ct)

    def _materialize_records(self, cfg):
        """Mutate cfg to add _records arrays for each source + metrics."""
        for src in cfg.get("rollout_sources", []):
            try:
                src["_records"] = load_source_records(src)
            except Exception as e:
                src["_records"] = []
                src["_error"] = str(e)
        if cfg.get("metrics"):
            try:
                cfg["metrics"]["_records"] = load_metrics_records(cfg["metrics"])
            except Exception as e:
                cfg["metrics"]["_records"] = []
                cfg["metrics"]["_error"] = str(e)


# ----------------------------------------------------------------------------
# Startup
# ----------------------------------------------------------------------------

def find_port():
    import socket
    for offset in range(PORT_TRIES):
        port = PORT_BASE + offset
        s = socket.socket()
        try:
            s.bind(("localhost", port))
            s.close()
            return port
        except OSError:
            continue
    return None


def print_summary(cfg, port):
    print(f"\ntraining-explorer → http://localhost:{port}\n")
    if cfg.get("_error"):
        print(f"  ⚠ {cfg['_error']}")
        return
    sources = cfg.get("rollout_sources", [])
    if sources:
        print(f"  rollout sources ({len(sources)}):")
        for s in sources:
            print(f"    · {s['name']} — {s.get('format', '?')} · "
                  f"{(s.get('n_rollouts') or 0):,} rollouts · {s.get('file', s.get('glob', '?'))}")
    else:
        print("  no rollout sources detected")
    if cfg.get("metrics"):
        print(f"  metrics: {cfg['metrics']['file']} · "
              f"{len(cfg['metrics'].get('series', []))} series")
    if cfg.get("config"):
        print(f"  config:  {cfg['config']['file']} · "
              f"{len(cfg['config'].get('values', {}))} keys")
    if cfg.get("tfevents"):
        print(f"  tfevents: {len(cfg['tfevents'])} file(s) — "
              f"use `tensorboard --logdir .` to view")
    print()


def main():
    port = find_port()
    if port is None:
        print("Could not find an open port near 8765.")
        sys.exit(1)

    cfg = load_config()
    print_summary(cfg, port)

    server = ThreadingHTTPServer(("localhost", port), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()

    try:
        webbrowser.open(f"http://localhost:{port}")
    except Exception:
        pass

    print("Press Ctrl+C to stop.\n")
    try:
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        print("\nshutting down")
        server.shutdown()


if __name__ == "__main__":
    main()

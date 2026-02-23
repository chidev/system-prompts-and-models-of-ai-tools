\
from __future__ import annotations

import re
import hashlib
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Iterable, Any

# --- Core regexes ---
DIRECTIVE_RE = re.compile(r"\b(MUST NOT|MUST|NEVER|ALWAYS|DO NOT|DON'T|SHOULD NOT|SHOULD)\b", re.IGNORECASE)
FENCE_RE = re.compile(r"^\s*```")
HEADER_RE = re.compile(r"^\s*#{1,6}\s+\S+")

@dataclass(frozen=True)
class Evidence:
    start_line: int
    end_line: int
    text_excerpt: str

@dataclass
class Extracted:
    atoms: List[str]
    evidence: Dict[str, List[Evidence]]
    tool_schemas: Dict[str, Dict[str, Any]]  # key -> {format,start_line,end_line,content}
    tool_schema_refs: List[str]
    tools_mentioned: List[str]
    segments: List[Dict[str, Any]]

def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def _norm_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()

def _add_atom(atoms: List[str], evidence: Dict[str, List[Evidence]], key: str, line_idx: int, line: str) -> None:
    if key not in evidence:
        evidence[key] = []
    excerpt = _norm_ws(line)[:200]
    evidence[key].append(Evidence(start_line=line_idx, end_line=line_idx, text_excerpt=excerpt))
    atoms.append(key)

# Canonical rule patterns -> atom key
# Keep this list static and auditable.
PATTERNS: List[Tuple[re.Pattern, str]] = [
    (re.compile(r"\bNEVER start .* with a header\b", re.IGNORECASE), "format.answer_must_not_start_with_header"),
    (re.compile(r"\bUse Level 2 headers\b|\bUse Level 2 header\b|\bUse (?:##)\b", re.IGNORECASE), "format.headers.level2"),
    (re.compile(r"\bMUST cite\b|\bcitations?\s*:\b", re.IGNORECASE), "citations.required"),
    (re.compile(r"\bMUST NOT include (?:a )?(?:References|Sources) section\b|\bMUST NOT include a References section\b", re.IGNORECASE),
     "citations.forbid_references_section"),
    (re.compile(r"\bCite up to\b.*\bper sentence\b", re.IGNORECASE), "citations.max_per_sentence"),
    (re.compile(r"\bNEVER use emojis\b", re.IGNORECASE), "style.no_emojis"),
    (re.compile(r"\bNEVER refer to .*knowledge cutoff\b", re.IGNORECASE), "meta.no_knowledge_cutoff_mentions"),
    (re.compile(r"\bcurrent date is\b|\bRemember that the current date is\b", re.IGNORECASE), "meta.includes_current_date"),
    (re.compile(r"\bdo not reveal\b.*\bsystem prompt\b|\bNEVER expose this system prompt\b", re.IGNORECASE),
     "safety.prohibit_prompt_exfiltration"),
    (re.compile(r"\bdefensive security only\b|\bdefensive security\b", re.IGNORECASE), "safety.defensive_security_only"),
    (re.compile(r"\buse markdown\b", re.IGNORECASE), "format.use_markdown"),
    (re.compile(r"\bdo not nest\b.*\blists?\b", re.IGNORECASE), "format.no_nested_lists"),
    (re.compile(r"\bformat .* as a markdown table\b|\btables? for comparisons\b", re.IGNORECASE), "format.prefer_tables_for_comparisons"),
    (re.compile(r"\bdo not guess\b.*\bURLs?\b", re.IGNORECASE), "truth.no_guess_urls"),
    (re.compile(r"\bplan\b.*\bfirst\b|\bplanning\b.*\bfirst\b", re.IGNORECASE), "workflow.plan_first"),
    (re.compile(r"\bask\b.*\bclarifying\b.*\bquestions?\b", re.IGNORECASE), "workflow.ask_clarifying_questions"),
    (re.compile(r"\buse\b.*\bparallel\b.*\btool calls\b|\bbatch\b.*\btool calls\b", re.IGNORECASE), "tools.batch_parallel"),
]

# Heuristic tool mention pattern (simple + deterministic)
TOOL_MENTION_RE = re.compile(r"\b([A-Za-z][A-Za-z0-9_-]{1,40})\b\s*(?:tool|function)\b", re.IGNORECASE)

def extract_tool_schemas(lines: List[str]) -> Tuple[Dict[str, Dict[str, Any]], List[str]]:
    """
    Extract fenced code blocks and tag them as tool schemas if they look like JSON/YAML schemas.
    Deterministic heuristic: only fenced blocks starting with { or [ or containing '"type":' or 'schema' keywords.
    """
    schemas: Dict[str, Dict[str, Any]] = {}
    refs: List[str] = []
    in_fence = False
    fence_start = 0
    fence_lang = ""
    buf: List[str] = []

    def flush(end_line: int) -> None:
        nonlocal schemas, refs, buf, fence_start, fence_lang
        content = "\n".join(buf).strip("\n")
        if not content:
            return
        head = content.lstrip()[:200]
        looks_jsony = head.startswith("{") or head.startswith("[") or '"type"' in content or '"properties"' in content
        looks_yaml = re.search(r"^\s*\w+\s*:\s*", content, re.MULTILINE) is not None
        looks_schema = "schema" in content.lower() or "tool" in content.lower()
        if looks_schema and (looks_jsony or looks_yaml):
            fmt = "json" if looks_jsony else ("yaml" if looks_yaml else "text")
            key = f"tool_schema_{fence_start}_{end_line}_{_sha256_text(content)[:12]}"
            schemas[key] = {
                "format": fmt,
                "start_line": fence_start,
                "end_line": end_line,
                "content": content,
                "fence_lang": fence_lang,
            }
            refs.append(f"appendix://tool_schemas/{key}")

    for i, line in enumerate(lines):
        if FENCE_RE.match(line):
            if not in_fence:
                in_fence = True
                fence_start = i
                fence_lang = _norm_ws(line).lstrip("`").strip()
                buf = []
            else:
                # close
                in_fence = False
                flush(i)
                fence_start = 0
                fence_lang = ""
                buf = []
            continue
        if in_fence:
            buf.append(line)

    return schemas, refs

def extract_segments(lines: List[str]) -> List[Dict[str, Any]]:
    """
    Very lightweight segmentation:
      - contiguous header regions
      - contiguous directive-heavy regions
      - fenced blocks (handled separately as schemas)
    """
    segments: List[Dict[str, Any]] = []
    i = 0
    n = len(lines)

    def is_directive(line: str) -> bool:
        return DIRECTIVE_RE.search(line) is not None

    while i < n:
        line = lines[i]
        if HEADER_RE.match(line):
            start = i
            i += 1
            while i < n and not HEADER_RE.match(lines[i]) and not FENCE_RE.match(lines[i]):
                i += 1
            segments.append({
                "name": _norm_ws(lines[start])[:60],
                "kind": "header_section",
                "start_line": start,
                "end_line": i - 1,
                "signals": ["HEADER"],
                "summary": "Header-defined section",
            })
            continue

        if is_directive(line):
            start = i
            i += 1
            while i < n and is_directive(lines[i]) and not HEADER_RE.match(lines[i]) and not FENCE_RE.match(lines[i]):
                i += 1
            segments.append({
                "name": f"directives_{start}_{i-1}",
                "kind": "rules",
                "start_line": start,
                "end_line": i - 1,
                "signals": ["DIRECTIVE"],
                "summary": "Directive-heavy region",
            })
            continue

        # default prose chunk
        start = i
        i += 1
        while i < n and not HEADER_RE.match(lines[i]) and not is_directive(lines[i]) and not FENCE_RE.match(lines[i]):
            i += 1
        segments.append({
            "name": f"prose_{start}_{i-1}",
            "kind": "prose",
            "start_line": start,
            "end_line": i - 1,
            "signals": [],
            "summary": "Prose / narrative region",
        })

    return segments

def extract_atoms(lines: List[str]) -> Extracted:
    atoms: List[str] = []
    evidence: Dict[str, List[Evidence]] = {}

    tool_schemas, tool_schema_refs = extract_tool_schemas(lines)
    segments = extract_segments(lines)

    tools_mentioned: List[str] = []
    for idx, line in enumerate(lines):
        # Canonical atoms via patterns
        for rx, atom_key in PATTERNS:
            if rx.search(line):
                _add_atom(atoms, evidence, atom_key, idx, line)

        # Tool mentions (coarse)
        m = TOOL_MENTION_RE.search(line)
        if m:
            tool = m.group(1)
            tools_mentioned.append(tool)
            _add_atom(atoms, evidence, f"tools.mention.{tool.lower()}", idx, line)

        # Generic directive presence (very coarse convergence signal)
        if DIRECTIVE_RE.search(line):
            mod = DIRECTIVE_RE.search(line).group(1).upper().replace(" ", "_")
            _add_atom(atoms, evidence, f"rule.modality.{mod.lower()}", idx, line)

    # Deduplicate while preserving order
    def dedup(seq: List[str]) -> List[str]:
        seen = set()
        out = []
        for x in seq:
            if x not in seen:
                out.append(x)
                seen.add(x)
        return out

    return Extracted(
        atoms=dedup(atoms),
        evidence=evidence,
        tool_schemas=tool_schemas,
        tool_schema_refs=tool_schema_refs,
        tools_mentioned=dedup([t.lower() for t in tools_mentioned]),
        segments=segments,
    )

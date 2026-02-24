\
from __future__ import annotations

import argparse
import hashlib
import os
import re
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import yaml
import pandas as pd

from promptos_atoms import extract_atoms
from promptos_graph import PromptRecord, build_bipartite_graph, build_similarity_graph, export_graphml


BINARY_EXTS = {
    ".png",".jpg",".jpeg",".gif",".webp",".svg",".ico",".pdf",".zip",".gz",".tar",".tgz",".7z",".mp4",".mov",".avi",".mp3",".wav"
}

SKIP_DIR_NAMES = {
    ".git",
    ".hg",
    ".svn",
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
    ".venv",
    "venv",
    "node_modules",
    "out",
}

def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

def load_yaml(path: Path) -> Dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))

def normalize_atoms(strict_atoms: List[str], ontology: Dict[str, Any]) -> List[str]:
    aliases = ontology.get("aliases", {}) or {}
    regex_aliases = ontology.get("regex_aliases", []) or []

    out: List[str] = []
    for a in strict_atoms:
        out.append(aliases.get(a, a))

    # Apply regex aliases (deterministic, order matters)
    normalized2: List[str] = []
    for a in out:
        replaced = a
        for rule in regex_aliases:
            pat = re.compile(rule["pattern"])
            if pat.search(replaced):
                replaced = pat.sub(rule["replace"], replaced)
        normalized2.append(replaced)

    # Dedup preserve order
    seen = set()
    deduped = []
    for a in normalized2:
        if a not in seen:
            deduped.append(a)
            seen.add(a)
    return deduped

def infer_product_vendor(path: str) -> Tuple[str, str]:
    # Repo folder name typically encodes product.
    parts = Path(path).parts
    product = parts[0] if parts else ""
    vendor = ""
    if product.lower() in {"anthropic", "google"}:
        vendor = product
    if "cursor" in product.lower():
        vendor = "Anysphere"
    if "perplexity" in product.lower():
        vendor = "Perplexity"
    if "lovable" in product.lower():
        vendor = "Lovable"
    return product, vendor

def classify_prompt(path: str, text: str) -> Dict[str, Any]:
    p = path.lower()
    kind = "other"
    archetype = "other"
    runtime = "unknown"
    loop = "unknown"
    tags: List[str] = []

    if "prompt" in p and ("agent" in p or "system" in p):
        kind = "system"
    if "tools" in p:
        kind = "tool"
        tags.append("tooling")
    if "model" in p:
        kind = "model_config"
        tags.append("model")

    if re.search(r"\byou are\b", text[:400], re.IGNORECASE):
        tags.append("role_declared")

    if "vscode" in p or "cursor" in p or "ide" in text.lower():
        runtime = "ide"
    if "cli" in p or "terminal" in text.lower():
        runtime = "cli"
    if "browser" in text.lower() or "web" in p:
        runtime = "web"

    if "code" in text.lower() or "coding" in text.lower():
        archetype = "coding_agent"
    if "search" in text.lower() and "sources" in text.lower():
        archetype = "search_writer"

    if "loop" in text.lower() or "while" in text.lower():
        loop = "agentic_loop"
    else:
        loop = "multi_turn"

    # Highly citation formatted
    if "citations" in text.lower() and "references section" in text.lower():
        tags.append("citation-heavy")

    return {
        "prompt_kind": kind,
        "agent_archetype": archetype,
        "runtime_surface": runtime,
        "loop_style": loop,
        "tags": tags,
    }

def build_execution(atoms: List[str]) -> Dict[str, Any]:
    # Deterministic FSM scaffold: include optional states based on atoms.
    states = ["START"]
    transitions = []

    def has(prefix: str) -> bool:
        return any(a == prefix or a.startswith(prefix) for a in atoms)

    if "workflow.plan_first" in atoms:
        states.append("PLAN")
        transitions.append({"from":"START","to":"PLAN","on":"workflow.plan_first"})
    else:
        transitions.append({"from":"START","to":"GATHER","on":"default"})

    if "workflow.ask_clarifying_questions" in atoms:
        if "CLARIFY" not in states:
            states.append("CLARIFY")
        transitions.append({"from": states[-2], "to":"CLARIFY", "on":"needs_clarification"})

    if has("tools.mention.") or has("tools.batch_parallel") or has("tools.policy"):
        if "ACT" not in states:
            states.append("ACT")
        transitions.append({"from": states[-2], "to":"ACT", "on":"tool_required"})

    if "GATHER" not in states:
        states.append("GATHER")
    # Ensure GATHER precedes RESPOND if citations or sources implied
    if "citations.required" in atoms and "GATHER" in states and states[-1] != "GATHER":
        # no-op: scaffold already has GATHER
        pass

    if "RESPOND" not in states:
        states.append("RESPOND")
    transitions.append({"from": states[-2], "to":"RESPOND", "on":"ready_to_answer"})

    states.append("END")
    transitions.append({"from":"RESPOND","to":"END","on":"done"})

    fsm = {
        "states": list(dict.fromkeys(states)),  # stable order
        "start_state": "START",
        "terminal_states": ["END"],
        "transitions": transitions,
    }

    # Behavior tree scaffold
    children = []
    children.append({"type":"leaf","name":"classify_request","children":[]})
    if "workflow.plan_first" in atoms:
        children.append({"type":"leaf","name":"make_plan","children":[]})
    if "workflow.ask_clarifying_questions" in atoms:
        children.append({"type":"decorator","name":"if_needs_clarification","children":[
            {"type":"leaf","name":"ask_clarifying_questions","children":[]}
        ]})
    if any(a.startswith("tools.") for a in atoms):
        children.append({"type":"decorator","name":"if_tools_needed","children":[
            {"type":"leaf","name":"select_and_call_tools","children":[]}
        ]})
    children.append({"type":"leaf","name":"compose_response","children":[]})

    bt = {"root":{"type":"sequence","name":"root_sequence","children":children}}

    return {"fsm": fsm, "behavior_tree": bt}

def compile_file(repo_url: str, ref: str, file_path: Path, rel_path: str, ontology: Dict[str, Any]) -> Dict[str, Any]:
    raw_bytes = file_path.read_bytes()
    try:
        text = raw_bytes.decode("utf-8")
    except UnicodeDecodeError:
        # fall back, lossy
        text = raw_bytes.decode("utf-8", errors="replace")

    lines = text.splitlines()
    extracted = extract_atoms(lines)

    strict_atoms = extracted.atoms
    normalized_atoms = normalize_atoms(strict_atoms, ontology)

    product, vendor = infer_product_vendor(rel_path)
    classification = classify_prompt(rel_path, text)

    prompt_id = sha256_bytes((repo_url + "::" + ref + "::" + rel_path).encode("utf-8"))

    dsl = {
        "version": 0.2,
        "prompt": {
            "id": prompt_id,
            "source": {
                "repo_url": repo_url,
                "ref": ref,
                "path": rel_path.replace(os.sep, "/"),
                "retrieved_at": datetime.now(timezone.utc).isoformat(),
            },
            "metadata": {
                "product": product,
                "vendor": vendor,
                "file_type": file_path.suffix.lstrip("."),
                "language_hint": "unknown",
                "date_hint": "",  # optional: could parse from text
                "size_bytes": len(raw_bytes),
                "sha256": sha256_bytes(raw_bytes),
                "notes": [],
            },
        },
        "classification": classification,
        "structure": {"segments": extracted.segments},
        "atoms": {
            "strict": strict_atoms,
            "normalized": normalized_atoms,
            "evidence": {
                k: [e.__dict__ for e in v] for k, v in extracted.evidence.items()
            },
        },
        "execution": build_execution(normalized_atoms),
        "toolcalling": {
            "tools_mentioned": extracted.tools_mentioned,
            "tool_policies": [],  # could be expanded deterministically later
            "tool_schemas_ref": extracted.tool_schema_refs,
        },
        "output_contract": {
            "tone": "unknown",
            "formatting": {
                "constraints": [],
                "citations": {
                    "required": "citations.required" in normalized_atoms,
                    "style": "unknown",
                    "max_per_sentence": 0,
                    "forbid_references_section": "citations.forbid_references_section" in normalized_atoms,
                },
            },
        },
        "safety": {
            "domains": [a for a in normalized_atoms if a.startswith("safety.")],
            "refusal_style": "unknown",
            "privacy": {
                "prohibit_prompt_exfiltration": "safety.prohibit_prompt_exfiltration" in normalized_atoms,
                "prohibit_user_data_leakage": False,
            },
        },
        "appendix": {
            "tool_schemas": extracted.tool_schemas,
            "raw_blocks": [],
        },
    }
    return dsl

def should_include(path: Path) -> bool:
    if path.suffix.lower() in BINARY_EXTS:
        return False
    if path.name.lower() in {"license", "license.md", "readme.md"}:
        # still useful but not a "system prompt"; keep out by default
        return False
    return True

def scan_files(repo_root: Path, out_dir: Optional[Path] = None) -> List[Path]:
    files: List[Path] = []
    skip_dirs = set(SKIP_DIR_NAMES)

    if out_dir is not None:
        out_resolved = out_dir.resolve()
        try:
            rel_out = out_resolved.relative_to(repo_root)
        except ValueError:
            rel_out = None
        if rel_out is not None and rel_out.parts:
            skip_dirs.add(rel_out.parts[0])

    for root, dirs, filenames in os.walk(repo_root):
        dirs[:] = [d for d in dirs if d not in skip_dirs]
        root_path = Path(root)
        for name in filenames:
            p = root_path / name
            if should_include(p):
                files.append(p)
    return sorted(files)

def write_yaml(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(obj, sort_keys=False, allow_unicode=True), encoding="utf-8")

def build_reports(out_dir: Path, dsls: List[Dict[str, Any]], ontology: Dict[str, Any]) -> None:
    # Index
    rows = []
    for d in dsls:
        rows.append({
            "prompt_id": d["prompt"]["id"],
            "path": d["prompt"]["source"]["path"],
            "product": d["prompt"]["metadata"]["product"],
            "vendor": d["prompt"]["metadata"]["vendor"],
            "prompt_kind": d["classification"]["prompt_kind"],
            "agent_archetype": d["classification"]["agent_archetype"],
            "runtime_surface": d["classification"]["runtime_surface"],
            "n_atoms_strict": len(d["atoms"]["strict"]),
            "n_atoms_normalized": len(d["atoms"]["normalized"]),
        })
    index_df = pd.DataFrame(rows).sort_values(["product","path"])
    (out_dir / "compiled").mkdir(parents=True, exist_ok=True)
    index_df.to_csv(out_dir / "compiled" / "index.csv", index=False)

    # Frequency + convergence summary
    def freq(atoms_key: str) -> pd.DataFrame:
        from collections import Counter
        c = Counter()
        for d in dsls:
            c.update(d["atoms"][atoms_key])
        df = pd.DataFrame([{"atom":k, "count":v} for k,v in c.items()]).sort_values("count", ascending=False)
        return df

    metrics_dir = out_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    freq("strict").to_csv(metrics_dir / "atom_freq_strict.csv", index=False)
    freq("normalized").to_csv(metrics_dir / "atom_freq_normalized.csv", index=False)

    # Graphs
    def export_view(view_name: str, prefixes: List[str], atoms_key: str, subdir: Path) -> None:
        records = []
        for d in dsls:
            atoms = [a for a in d["atoms"][atoms_key] if any(a.startswith(px) for px in prefixes)]
            records.append(PromptRecord(prompt_id=d["prompt"]["id"], label=d["prompt"]["source"]["path"], atoms=atoms))
        bip = build_bipartite_graph(records, graph_name=f"{view_name}_bipartite_{atoms_key}")
        sim = build_similarity_graph(records, graph_name=f"{view_name}_similarity_{atoms_key}")
        export_graphml(bip, str(subdir / f"{view_name}_bipartite_{atoms_key}.graphml"))
        export_graphml(sim, str(subdir / f"{view_name}_similarity_{atoms_key}.graphml"))

    graphs_dir = out_dir / "graphs"
    graphs_dir.mkdir(parents=True, exist_ok=True)

    # Overall graphs
    for atoms_key in ["strict","normalized"]:
        records = [PromptRecord(prompt_id=d["prompt"]["id"], label=d["prompt"]["source"]["path"], atoms=d["atoms"][atoms_key]) for d in dsls]
        export_graphml(build_bipartite_graph(records, graph_name=f"all_bipartite_{atoms_key}"), str(graphs_dir / f"all_bipartite_{atoms_key}.graphml"))
        export_graphml(build_similarity_graph(records, graph_name=f"all_similarity_{atoms_key}"), str(graphs_dir / f"all_similarity_{atoms_key}.graphml"))

        views = ontology.get("views", {}) or {}
        for view_name, prefixes in views.items():
            export_view(view_name, prefixes, atoms_key, graphs_dir)

def main() -> None:
    ap = argparse.ArgumentParser(description="Compile prompt artifacts into PromptOS DSL and graphs.")
    ap.add_argument("--repo", required=True, help="Path to local clone of system-prompts-and-models-of-ai-tools")
    ap.add_argument("--out", required=True, help="Output directory")
    ap.add_argument("--repo-url", default="https://github.com/chidev/system-prompts-and-models-of-ai-tools")
    ap.add_argument("--ref", default="main")
    ap.add_argument("--ontology", default=str(Path(__file__).with_name("ontology_v0.1.yaml")))
    args = ap.parse_args()

    repo_root = Path(args.repo).resolve()
    out_dir = Path(args.out).resolve()
    ontology = load_yaml(Path(args.ontology))

    files = scan_files(repo_root, out_dir)

    dsls: List[Dict[str, Any]] = []
    for f in files:
        rel_path = str(f.relative_to(repo_root))
        dsl = compile_file(args.repo_url, args.ref, f, rel_path, ontology)
        dsls.append(dsl)

        out_path = out_dir / "compiled" / (rel_path.replace(os.sep, "__") + ".yaml")
        write_yaml(out_path, dsl)

    build_reports(out_dir, dsls, ontology)

if __name__ == "__main__":
    main()

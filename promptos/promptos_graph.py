\
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Iterable, Optional, Set
import networkx as nx

@dataclass(frozen=True)
class PromptRecord:
    prompt_id: str
    label: str
    atoms: List[str]

def jaccard(a: Set[str], b: Set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0

def build_bipartite_graph(records: List[PromptRecord], graph_name: str = "prompts_features") -> nx.Graph:
    G = nx.Graph()
    G.graph["name"] = graph_name
    for rec in records:
        G.add_node(rec.prompt_id, kind="prompt", label=rec.label)
        for atom in rec.atoms:
            atom_id = f"atom::{atom}"
            if not G.has_node(atom_id):
                G.add_node(atom_id, kind="atom", label=atom)
            G.add_edge(rec.prompt_id, atom_id)
    return G

def build_similarity_graph(records: List[PromptRecord], graph_name: str = "prompts_similarity") -> nx.Graph:
    G = nx.Graph()
    G.graph["name"] = graph_name
    for rec in records:
        G.add_node(rec.prompt_id, kind="prompt", label=rec.label)

    atoms_by_id = {r.prompt_id: set(r.atoms) for r in records}
    ids = [r.prompt_id for r in records]
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            a, b = ids[i], ids[j]
            sim = jaccard(atoms_by_id[a], atoms_by_id[b])
            if sim > 0:
                G.add_edge(a, b, weight=sim)
    return G

def export_graphml(G: nx.Graph, path: str) -> None:
    nx.write_graphml(G, path)

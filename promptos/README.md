PromptOS Compiler (deterministic)

What this does
- Scans a local clone of system-prompts-and-models-of-ai-tools
- Compiles each artifact into PromptOS DSL (YAML)
- Extracts deterministic "atoms" for overlap detection
- Emits multiple graph views (strict vs normalized; tools/workflow/output/safety/meta)

Design choices (aligned to your constraints)
- Tool calling behavior is first-class and becomes atoms (appears in overlap graphs).
- Tool schemas (long JSON/YAML blocks) do NOT dominate the main graph: they are stored in appendix.tool_schemas and referenced from toolcalling.tool_schemas_ref.
- Strict vs normalized are treated as two separate graph views. Normalization is fully deterministic via ontology_v0.1.yaml.

Quick start
1) Clone the repo locally (any fork is fine):
   git clone https://github.com/chidev/system-prompts-and-models-of-ai-tools.git

2) Run the compiler:
   python promptos_compile.py --repo ./system-prompts-and-models-of-ai-tools --out ./out

Outputs
- out/compiled/index.csv                         : index over all compiled artifacts
- out/compiled/<relpath__file>.yaml              : DSL per file
- out/metrics/atom_freq_{strict,normalized}.csv  : convergence frequencies
- out/graphs/all_*_{strict,normalized}.graphml   : full bipartite + similarity graphs
- out/graphs/{tools,workflow,output,safety,meta}_*_{strict,normalized}.graphml : view graphs

Graph usage
- GraphML loads cleanly in Gephi and Cytoscape.
- Use all_similarity_* to see prompt families (edge weight = Jaccard over atoms).
- Use all_bipartite_* to see which atoms drive convergence (features with high degree).

Extending atoms (how to evolve without losing determinism)
- Add a new regex -> atom mapping in promptos_atoms.PATTERNS
- Optionally add alias mappings in ontology_v0.1.yaml
- Keep both changes static and version-controlled.

Known limitations (intentional for v0)
- Segmenter and classifier are heuristic and shallow.
- Tool schema extraction only captures fenced blocks.
- Many directives will be captured only as coarse modality atoms (rule.modality.*) until you add richer canonical patterns.


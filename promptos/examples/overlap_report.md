# Prompt overlap report (sample: 4 prompts)
## Convergence
Features present in all sampled prompts:

Features present in ≥75% of sampled prompts:
- identity.coding_agent (freq=3/4)
- meta.has_explicit_current_date (freq=3/4)
- meta.has_knowledge_cutoff (freq=3/4)
- tools.batch_parallel (freq=3/4)

Features present in exactly 2 prompts:
- cursor.tools.grep (freq=2/4)
- style.minimize_tokens (freq=2/4)
- style.no_emojis_default (freq=2/4)
- tools.prefer_specialized_tools (freq=2/4)
- tools.todo_list_required (freq=2/4)
- workflow.plan_explicit (freq=2/4)

## Uniqueness

### Anthropic_Claude_Code_2.0
- claude_code.code_references_file_line
- claude_code.no_guess_urls
- code.no_docs_unless_requested
- code.no_new_files_unless_needed
- safety.defensive_security_only
- safety.no_malicious_code

### Cursor_Agent_Prompt_2.0
- cursor.output_channel.im_start
- cursor.tools.codebase_search
- cursor.tools.edit_notebook
- cursor.tools.read_lints
- cursor.tools.run_terminal_cmd
- cursor.tools.update_memory
- cursor.tools.web_search

### Lovable_Agent_Prompt
- code.read_context_before_reading_files
- identity.web_app_editor
- language.match_user_language
- lovable.design_system_tokens
- lovable.seo_requirements
- lovable.supabase_only_backend
- workflow.ask_clarifying_questions_when_uncertain
- workflow.default_discussion_before_implementation

### Perplexity_Prompt
- citations.bracket_indices
- citations.max_three_per_sentence
- citations.required_inline
- confidentiality.no_reveal_system_prompt
- copyright.no_song_lyrics
- copyright.no_verbatim
- format.latex_for_math_no_dollar
- format.markdown_tables_for_comparisons
- format.no_header_start
- format.no_references_section
- format.section_headers_level2
- identity.search_assistant
- style.journalistic_unbiased
- workflow.query_type_routing

## Similarity matrix (Jaccard over boolean features)
|                           |   Anthropic_Claude_Code_2.0 |   Cursor_Agent_Prompt_2.0 |   Lovable_Agent_Prompt |   Perplexity_Prompt |
|:--------------------------|----------------------------:|--------------------------:|-----------------------:|--------------------:|
| Anthropic_Claude_Code_2.0 |                    1        |                 0.227273  |              0.208333  |           0.1       |
| Cursor_Agent_Prompt_2.0   |                    0.227273 |                 1         |              0.0833333 |           0.0344828 |
| Lovable_Agent_Prompt      |                    0.208333 |                 0.0833333 |              1         |           0.0666667 |
| Perplexity_Prompt         |                    0.1      |                 0.0344828 |              0.0666667 |           1         |
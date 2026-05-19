# Pipeline Health

- Bundle valid: `True`
- Ready for scale: `False`
- Completed runs: `224/224`
- Running runs: `0`
- Total runs: `224`
- Trace coverage: `100.0%`
- Mean invalid action outputs: `1.69`
- Mean action-output retries: `1.67`
- Mean provider tool-call fallbacks: `0.00`
- Mean invalid model outputs: `1.69`
- Mean model-output retries: `1.67`
- Mean model-output format errors: `1.54`
- Mean grounding errors: `0.16`
- Mean prompt tokens: `73076.8`
- Mean artifact-integrity errors: `0.02`
- Mean post-completion turns: `0.00`
- Max single-turn prompt tokens: `42246`
- Max single-turn completion tokens: `6144`

## Blocking Reasons

- 4 run(s) recorded artifact-integrity errors.
- At least one run recorded provider errors.
- At least one run recorded a runtime error.
- At least one run ended in runner_failure.
- Mean invalid action outputs exceeded the scale limit (1.0).
- Mean action-output retries exceeded the scale limit (1.0).

## Warnings

- 10 runs exceeded the prompt-token warning band (200000).
- 25 runs fell below the completion ratio floor (0.2).
- 4 runs recorded artifact-integrity issues.

## By Condition

- `global_one_shot_planner`: invalid_action_outputs=0.39, retries=0.34, provider_tool_call_fallbacks=0.00, invalid_model_outputs=0.39, grounding_errors=0.11, prompt_tokens=13748.6, completion=0.77
- `global_structured_interactive_grounded`: invalid_action_outputs=2.64, retries=2.62, provider_tool_call_fallbacks=0.00, invalid_model_outputs=2.64, grounding_errors=0.20, prompt_tokens=120778.1, completion=0.62
- `local_narrative_interactive_grounded`: invalid_action_outputs=1.41, retries=1.38, provider_tool_call_fallbacks=0.00, invalid_model_outputs=1.41, grounding_errors=0.11, prompt_tokens=71144.3, completion=0.70
- `local_structured_interactive_grounded`: invalid_action_outputs=2.32, retries=2.32, provider_tool_call_fallbacks=0.00, invalid_model_outputs=2.32, grounding_errors=0.21, prompt_tokens=86636.1, completion=0.61

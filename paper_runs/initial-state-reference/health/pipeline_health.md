# Pipeline Health

- Bundle valid: `True`
- Ready for scale: `False`
- Completed runs: `224/224`
- Running runs: `0`
- Total runs: `224`
- Trace coverage: `100.0%`
- Mean invalid action outputs: `1.73`
- Mean action-output retries: `1.71`
- Mean provider tool-call fallbacks: `0.00`
- Mean invalid model outputs: `1.73`
- Mean model-output retries: `1.71`
- Mean model-output format errors: `1.64`
- Mean grounding errors: `0.09`
- Mean prompt tokens: `69406.4`
- Mean artifact-integrity errors: `0.02`
- Mean post-completion turns: `0.00`
- Max single-turn prompt tokens: `38628`
- Max single-turn completion tokens: `6144`

## Blocking Reasons

- 5 run(s) recorded artifact-integrity errors.
- At least one run recorded provider errors.
- At least one run recorded a runtime error.
- At least one run ended in runner_failure.
- Mean invalid action outputs exceeded the scale limit (1.0).
- Mean action-output retries exceeded the scale limit (1.0).

## Warnings

- 8 runs exceeded the prompt-token warning band (200000).
- 24 runs fell below the completion ratio floor (0.2).
- 5 runs recorded artifact-integrity issues.

## By Condition

- `global_one_shot_planner`: invalid_action_outputs=0.32, retries=0.30, provider_tool_call_fallbacks=0.00, invalid_model_outputs=0.32, grounding_errors=0.09, prompt_tokens=12076.5, completion=0.84
- `global_structured_interactive_grounded`: invalid_action_outputs=2.84, retries=2.80, provider_tool_call_fallbacks=0.00, invalid_model_outputs=2.84, grounding_errors=0.07, prompt_tokens=109139.3, completion=0.58
- `local_narrative_interactive_grounded`: invalid_action_outputs=2.04, retries=2.02, provider_tool_call_fallbacks=0.00, invalid_model_outputs=2.04, grounding_errors=0.12, prompt_tokens=78980.1, completion=0.75
- `local_structured_interactive_grounded`: invalid_action_outputs=1.73, retries=1.70, provider_tool_call_fallbacks=0.00, invalid_model_outputs=1.73, grounding_errors=0.09, prompt_tokens=77429.7, completion=0.61

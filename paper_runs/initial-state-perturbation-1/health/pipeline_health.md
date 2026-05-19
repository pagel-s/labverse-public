# Pipeline Health

- Bundle valid: `True`
- Ready for scale: `False`
- Completed runs: `224/224`
- Running runs: `0`
- Total runs: `224`
- Trace coverage: `100.0%`
- Mean invalid action outputs: `1.69`
- Mean action-output retries: `1.62`
- Mean provider tool-call fallbacks: `0.00`
- Mean invalid model outputs: `1.69`
- Mean model-output retries: `1.62`
- Mean model-output format errors: `1.54`
- Mean grounding errors: `0.16`
- Mean prompt tokens: `72766.4`
- Mean artifact-integrity errors: `0.02`
- Mean post-completion turns: `0.00`
- Max single-turn prompt tokens: `41937`
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
- 25 runs fell below the completion ratio floor (0.2).
- 5 runs recorded artifact-integrity issues.

## By Condition

- `global_one_shot_planner`: invalid_action_outputs=0.39, retries=0.30, provider_tool_call_fallbacks=0.00, invalid_model_outputs=0.39, grounding_errors=0.18, prompt_tokens=13274.0, completion=0.78
- `global_structured_interactive_grounded`: invalid_action_outputs=2.57, retries=2.54, provider_tool_call_fallbacks=0.00, invalid_model_outputs=2.57, grounding_errors=0.21, prompt_tokens=120952.6, completion=0.65
- `local_narrative_interactive_grounded`: invalid_action_outputs=1.77, retries=1.70, provider_tool_call_fallbacks=0.00, invalid_model_outputs=1.77, grounding_errors=0.16, prompt_tokens=77061.4, completion=0.66
- `local_structured_interactive_grounded`: invalid_action_outputs=2.04, retries=1.96, provider_tool_call_fallbacks=0.00, invalid_model_outputs=2.04, grounding_errors=0.07, prompt_tokens=79777.6, completion=0.63

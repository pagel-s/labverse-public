# Inspect Traces

Start the paper-inspection API from the repository root:

```bash
labverse paper-serve --host 127.0.0.1 --port 8001
```

Start the frontend in another terminal:

```bash
cd frontend
npm ci
npm run dev
```

Open:

```text
http://127.0.0.1:5173
```

The viewer defaults to:

```text
paper_runs/initial-state-perturbation-2
```

Use the bundle preset buttons to switch between:

```text
Reference state
Perturbation 1
Perturbation 2
```

Select a run in the left rail, then step through the trace. Each step exposes
the recorded system prompt, user message, model attempts, resolved action,
action result, raw JSON, and replay frame before/after the action.

The public API intentionally exposes only:

```text
/health
/api/paper/status
/api/paper/runs
/api/paper/steps
/api/paper/step
/api/paper/chat-view
```

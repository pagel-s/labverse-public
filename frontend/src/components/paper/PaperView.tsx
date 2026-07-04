/**
 * PaperView — frozen run bundle inspector.
 *
 * Three-region studio for paper reviewers:
 *   • Rail   — bundle status, run list
 *   • Stage  — replay specimen (Phaser canvas), filmstrip step navigator,
 *              "what changed" delta summary
 *   • Trace  — system prompt, user message, assistant attempts, action
 *              result, raw step JSON. Structured records render through
 *              JsonTree; long-form prompts stay as monospace bodies.
 *
 * Keyboard navigation while a run is loaded:
 *   ←  /  k     previous step
 *   →  /  j     next step
 *   Home / End  jump to first / last
 *   b           replay world BEFORE the action
 *   a           replay world AFTER the action
 */

import { lazy, Suspense, useEffect, useMemo, useRef, useState } from "react";

import {
  apiClient,
  type PaperBundleStatus,
  type PaperRunSummary,
  type PaperStepDetail,
  type PaperStepSummary,
} from "../../api/client";

import JsonTree from "./JsonTree";
import type { PhaserGameHandle } from "../labville/PhaserGame";

const PhaserGame = lazy(() => import("../labville/PhaserGame"));

const LAST_BUNDLE_KEY = "labverse.paper.bundleDir";
const DEFAULT_BUNDLE_DIR =
  import.meta.env.VITE_LABVERSE_DEFAULT_BUNDLE_DIR ??
  "paper_runs/initial-state-perturbation-2";
const BUNDLE_PRESETS = [
  {
    label: "Perturbation 2",
    path: "paper_runs/initial-state-perturbation-2",
  },
  {
    label: "Reference state",
    path: "paper_runs/initial-state-reference",
  },
  {
    label: "Perturbation 1",
    path: "paper_runs/initial-state-perturbation-1",
  },
] as const;

type ReplayMode = "before" | "after";

interface AttemptRecord {
  attempt_index?: number;
  response_tool_call?: Record<string, unknown> | null;
  response_text?: string | null;
  invalid_reason?: string | null;
  provider_error?: string | null;
  metadata?: Record<string, unknown>;
}

function initialBundlePath(): string {
  if (typeof window === "undefined") {
    return DEFAULT_BUNDLE_DIR;
  }
  const queryBundle = new URLSearchParams(window.location.search).get("bundle");
  if (queryBundle) {
    return queryBundle;
  }
  const saved = window.localStorage.getItem(LAST_BUNDLE_KEY);
  if (!saved || saved.startsWith("/tmp/")) {
    window.localStorage.setItem(LAST_BUNDLE_KEY, DEFAULT_BUNDLE_DIR);
    return DEFAULT_BUNDLE_DIR;
  }
  return saved;
}

function statusTone(status: string | null): "ready" | "running" | "attention" | "idle" {
  if (status === "completed") return "ready";
  if (status === "running") return "running";
  if (status === "failed") return "attention";
  return "idle";
}

function stepTone(status: string | null): "ready" | "running" | "attention" | "idle" {
  if (status === "succeeded" || status === "executed") return "ready";
  if (status === "failed") return "attention";
  if (status === "pending") return "running";
  return "idle";
}

function actionPhase(
  detail: PaperStepDetail | null,
): "executing" | "resolved" | "blocked" | "failed" | null {
  if (!detail) return null;
  const step = detail.step as Record<string, unknown>;
  const actionResult = step.action_result as Record<string, unknown> | undefined;
  const actionStatus = typeof actionResult?.status === "string" ? actionResult.status : null;
  if (actionStatus === "succeeded") return "resolved";
  if (actionStatus === "blocked") return "blocked";
  if (actionStatus === "failed") return "failed";
  const status = typeof step.status === "string" ? step.status : null;
  if (status === "failed") return "failed";
  if (status === "pending") return "executing";
  return "resolved";
}

function shortRunLabel(run: PaperRunSummary): string {
  if (run.scenario && run.replication !== null) {
    return `${run.scenario} · rep ${run.replication}`;
  }
  return run.scenario ?? run.run_path.split("/").pop() ?? run.run_path;
}

function asString(value: unknown): string {
  if (value === null || value === undefined) return "";
  if (typeof value === "string") return value;
  return String(value);
}

function stepLabel(step: { resolved_candidate_ref?: string | null; resolved_verb?: string | null; resolved_action_id?: string | null; prompt_surface_kind?: string | null }): string {
  return step.resolved_candidate_ref ?? step.resolved_verb ?? step.resolved_action_id ?? step.prompt_surface_kind ?? "step";
}

function captureConfig(): {
  bundleDir: string;
  runPath: string | null;
  stepIndex: number | null;
  replayMode: ReplayMode;
  autoplay: boolean;
} | null {
  if (typeof window === "undefined") return null;
  const params = new URLSearchParams(window.location.search);
  if (params.get("capture") !== "1") return null;
  const requestedStep = Number.parseInt(params.get("step") ?? "", 10);
  return {
    bundleDir: params.get("bundle") ?? DEFAULT_BUNDLE_DIR,
    runPath: params.get("run"),
    stepIndex: Number.isFinite(requestedStep) ? requestedStep : null,
    replayMode: params.get("mode") === "before" ? "before" : "after",
    autoplay: params.get("autoplay") === "1",
  };
}

function captureCaption(
  stepIndex: number | null,
  totalSteps: number,
  detail: PaperStepDetail | null,
): { title: string; body: string } | null {
  if (stepIndex === null || !detail) return null;
  const step = detail.step as Record<string, unknown>;
  const actionResult = step.action_result as Record<string, unknown> | undefined;
  const description =
    typeof actionResult?.description === "string"
      ? actionResult.description
      : detail.replay_frame?.change_summary[0] ?? null;
  const label = stepLabel({
    resolved_candidate_ref: detail.replay_frame?.resolved_candidate_ref ?? null,
    resolved_verb: detail.replay_frame?.resolved_verb ?? null,
    resolved_action_id: detail.replay_frame?.resolved_action_id ?? null,
    prompt_surface_kind:
      typeof step.prompt_surface_kind === "string" ? step.prompt_surface_kind : null,
  });
  return {
    title: `Step ${stepIndex}/${totalSteps} · ${label.replaceAll("_", " ")}`,
    body: description ?? "Recorded laboratory state transition.",
  };
}

export default function PaperView(): JSX.Element {
  const capture = useMemo(() => captureConfig(), []);
  const [bundleDir, setBundleDir] = useState(initialBundlePath);
  const [loadedBundleDir, setLoadedBundleDir] = useState(initialBundlePath);
  const [status, setStatus] = useState<PaperBundleStatus | null>(null);
  const [runs, setRuns] = useState<PaperRunSummary[]>([]);
  const [selectedRun, setSelectedRun] = useState<string | null>(null);
  const [steps, setSteps] = useState<PaperStepSummary[]>([]);
  const [selectedStep, setSelectedStep] = useState<number | null>(null);
  const [stepDetail, setStepDetail] = useState<PaperStepDetail | null>(null);
  const [loadingBundle, setLoadingBundle] = useState(false);
  const [loadingSteps, setLoadingSteps] = useState(false);
  const [loadingDetail, setLoadingDetail] = useState(false);
  const [savingScene, setSavingScene] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [replayMode, setReplayMode] = useState<ReplayMode>(
    capture?.autoplay ? "before" : capture?.replayMode ?? "after",
  );
  const [capturedScene, setCapturedScene] = useState<string | null>(null);
  const [captureSceneReady, setCaptureSceneReady] = useState(false);
  const loadRunRequestId = useRef(0);
  const loadDetailRequestId = useRef(0);
  const filmstripRef = useRef<HTMLDivElement | null>(null);
  const phaserGameRef = useRef<PhaserGameHandle | null>(null);

  const selectedRunSummary = useMemo(
    () => runs.find((run) => run.run_path === selectedRun) ?? null,
    [runs, selectedRun],
  );
  const selectedStepSummary = useMemo(
    () => steps.find((step) => step.step_index === selectedStep) ?? null,
    [steps, selectedStep],
  );

  const stepDetailMatchesSelection =
    stepDetail !== null &&
    stepDetail.run_path === selectedRun &&
    stepDetail.step_index === selectedStep;

  const currentStep = stepDetailMatchesSelection
    ? (stepDetail.step as Record<string, unknown>)
    : null;
  const attempts = useMemo<AttemptRecord[]>(() => {
    if (!currentStep || !Array.isArray(currentStep.attempts)) {
      return [];
    }
    return currentStep.attempts as AttemptRecord[];
  }, [currentStep]);

  const replayFrame = stepDetailMatchesSelection ? stepDetail.replay_frame ?? null : null;
  const replayPhase = stepDetailMatchesSelection ? actionPhase(stepDetail) : null;
  const replayWorld =
    replayFrame === null
      ? null
      : replayMode === "before"
        ? replayFrame.world_before
        : replayFrame.world_after;
  const replayLoading =
    selectedRun !== null &&
    selectedStep !== null &&
    (loadingDetail || (stepDetail !== null && !stepDetailMatchesSelection));

  const replayFeedback =
    replayFrame && stepDetail && replayPhase
      ? {
          nonce: stepDetail.step_index,
          agentId: replayFrame.actor_id ?? "paper",
          verb: replayFrame.resolved_verb ?? replayFrame.resolved_candidate_ref ?? replayFrame.resolved_action_id ?? "paper_step",
          targetId: null,
          phase: replayPhase,
          outcome:
            (typeof (currentStep?.action_result as Record<string, unknown> | undefined)?.description === "string"
              ? ((currentStep?.action_result as Record<string, unknown>).description as string)
              : replayFrame.change_summary[0] ?? "Recorded paper step"),
        }
      : null;
  const caption = captureCaption(selectedStep, steps.length, stepDetail);

  const currentStepIndex = useMemo(
    () => (selectedStep === null ? -1 : steps.findIndex((s) => s.step_index === selectedStep)),
    [steps, selectedStep],
  );

  async function loadBundle(targetBundleDir: string): Promise<void> {
    const normalized = targetBundleDir.trim();
    if (!normalized) {
      setError("Enter a bundle directory first.");
      return;
    }
    setLoadingBundle(true);
    setError(null);
    try {
      await apiClient.getPaperHealth();
      const [nextStatus, nextRuns] = await Promise.all([
        apiClient.getPaperBundleStatus(normalized),
        apiClient.listPaperRuns(normalized),
      ]);
      const requestedRun =
        capture?.runPath && nextRuns.some((run) => run.run_path === capture.runPath)
          ? capture.runPath
          : null;
      const nextSelectedRun =
        requestedRun ??
        nextStatus.running_runs[0] ??
        nextStatus.latest_run ??
        nextRuns[0]?.run_path ??
        null;
      setStatus(nextStatus);
      setRuns(nextRuns);
      setSelectedRun(nextSelectedRun);
      setSteps([]);
      setSelectedStep(null);
      setStepDetail(null);
      setLoadedBundleDir(normalized);
      if (typeof window !== "undefined") {
        window.localStorage.setItem(LAST_BUNDLE_KEY, normalized);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load paper bundle.");
      setStatus(null);
      setRuns([]);
      setSteps([]);
      setStepDetail(null);
    } finally {
      setLoadingBundle(false);
    }
  }

  async function loadRunSteps(runPath: string): Promise<void> {
    if (!loadedBundleDir) return;
    const requestId = ++loadRunRequestId.current;
    setLoadingSteps(true);
    setError(null);
    setStepDetail(null);
    try {
      const nextSteps = await apiClient.listPaperSteps(loadedBundleDir, runPath);
      if (requestId !== loadRunRequestId.current) {
        return;
      }
      setSteps(nextSteps);
      const latestStep = nextSteps[nextSteps.length - 1]?.step_index ?? null;
      setSelectedStep((current) => {
        if (
          capture?.runPath === runPath &&
          capture.stepIndex !== null &&
          nextSteps.some((step) => step.step_index === capture.stepIndex)
        ) {
          return capture.stepIndex;
        }
        if (current && nextSteps.some((step) => step.step_index === current)) {
          return current;
        }
        return latestStep;
      });
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load recorded steps.");
      setSteps([]);
      setSelectedStep(null);
      setStepDetail(null);
    } finally {
      setLoadingSteps(false);
    }
  }

  async function loadStepDetail(runPath: string, stepIndex: number): Promise<void> {
    if (!loadedBundleDir) return;
    const requestId = ++loadDetailRequestId.current;
    setLoadingDetail(true);
    setError(null);
    setStepDetail(null);
    try {
      const nextDetail = await apiClient.getPaperStepDetail({
        bundleDir: loadedBundleDir,
        run: runPath,
        step: stepIndex,
      });
      if (requestId !== loadDetailRequestId.current) {
        return;
      }
      setStepDetail(nextDetail);
    } catch (err) {
      if (requestId !== loadDetailRequestId.current) {
        return;
      }
      setError(err instanceof Error ? err.message : "Failed to load step detail.");
      setStepDetail(null);
    } finally {
      if (requestId === loadDetailRequestId.current) {
        setLoadingDetail(false);
      }
    }
  }

  useEffect(() => {
    if (loadedBundleDir) {
      void loadBundle(loadedBundleDir);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    if (
      !capture ||
      capture.autoplay ||
      capturedScene ||
      !replayWorld ||
      !stepDetailMatchesSelection ||
      replayLoading
    ) {
      return;
    }
    const timer = window.setTimeout(() => {
      void phaserGameRef.current
        ?.captureScene()
        .then(setCapturedScene)
        .catch((captureError: unknown) => {
          setError(captureError instanceof Error ? captureError.message : "Could not capture the replay scene.");
        });
    }, 700);
    return () => window.clearTimeout(timer);
  }, [capture, capturedScene, replayWorld, replayLoading, stepDetailMatchesSelection]);

  useEffect(() => {
    if (!capture?.autoplay || steps.length <= 1 || currentStepIndex < 0) {
      return;
    }
    if (replayMode === "before") {
      const timer = window.setTimeout(() => {
        setReplayMode("after");
      }, 520);
      return () => window.clearTimeout(timer);
    }
    const timer = window.setTimeout(() => {
      const next = steps[(currentStepIndex + 1) % steps.length];
      if (next) {
        setReplayMode("before");
        setSelectedStep(next.step_index);
      }
    }, 1350);
    return () => window.clearTimeout(timer);
  }, [capture?.autoplay, steps, currentStepIndex, replayMode]);

  useEffect(() => {
    if (!capture) return;
    const timer = window.setInterval(() => {
      const canvas = document.querySelector<HTMLCanvasElement>(".paper-capture-stage canvas");
      setCaptureSceneReady(Boolean(canvas && canvas.width > 100 && canvas.height > 100));
    }, 100);
    return () => window.clearInterval(timer);
  }, [capture]);

  useEffect(() => {
    if (!capture) return;
    document.body.dataset.captureReady =
      replayWorld && stepDetailMatchesSelection && captureSceneReady ? "1" : "0";
    document.body.dataset.captureStep = selectedStep === null ? "" : String(selectedStep);
    return () => {
      delete document.body.dataset.captureReady;
      delete document.body.dataset.captureStep;
    };
  }, [capture, captureSceneReady, replayWorld, selectedStep, stepDetailMatchesSelection]);

  useEffect(() => {
    if (selectedRun) {
      void loadRunSteps(selectedRun);
    } else {
      loadRunRequestId.current += 1;
      loadDetailRequestId.current += 1;
      setSteps([]);
      setSelectedStep(null);
      setStepDetail(null);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedRun, loadedBundleDir]);

  useEffect(() => {
    if (selectedRun && selectedStep !== null) {
      void loadStepDetail(selectedRun, selectedStep);
    } else {
      loadDetailRequestId.current += 1;
      setStepDetail(null);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedRun, selectedStep, loadedBundleDir]);

  // Keyboard navigation. Skips when typing in inputs.
  useEffect(() => {
    function isEditableTarget(target: EventTarget | null): boolean {
      if (!(target instanceof HTMLElement)) return false;
      const tag = target.tagName.toLowerCase();
      if (tag === "input" || tag === "textarea" || tag === "select") return true;
      return target.isContentEditable;
    }

    function onKey(event: KeyboardEvent): void {
      if (event.metaKey || event.ctrlKey || event.altKey) return;
      if (isEditableTarget(event.target)) return;
      if (steps.length === 0) return;

      const idx = currentStepIndex;
      let nextIdx: number | null = null;

      if (event.key === "ArrowLeft" || event.key === "k") {
        nextIdx = idx <= 0 ? 0 : idx - 1;
      } else if (event.key === "ArrowRight" || event.key === "j") {
        nextIdx = idx < 0 ? 0 : Math.min(steps.length - 1, idx + 1);
      } else if (event.key === "Home") {
        nextIdx = 0;
      } else if (event.key === "End") {
        nextIdx = steps.length - 1;
      } else if (event.key === "b") {
        event.preventDefault();
        setReplayMode("before");
        return;
      } else if (event.key === "a") {
        event.preventDefault();
        setReplayMode("after");
        return;
      }

      if (nextIdx !== null && steps[nextIdx]) {
        event.preventDefault();
        setSelectedStep(steps[nextIdx].step_index);
      }
    }

    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [steps, currentStepIndex]);

  // Scroll the active filmstrip cell into view as the selection changes.
  useEffect(() => {
    if (!filmstripRef.current || selectedStep === null) return;
    const node = filmstripRef.current.querySelector<HTMLElement>(
      `[data-step-index="${selectedStep}"]`,
    );
    if (node) {
      node.scrollIntoView({ block: "nearest", inline: "center", behavior: "smooth" });
    }
  }, [selectedStep]);

  function gotoOffset(offset: number): void {
    if (currentStepIndex < 0) return;
    const next = steps[currentStepIndex + offset];
    if (next) setSelectedStep(next.step_index);
  }

  async function saveScene(): Promise<void> {
    if (!phaserGameRef.current || !selectedRunSummary || selectedStep === null) return;
    setSavingScene(true);
    try {
      const source = await phaserGameRef.current.captureScene();
      const scenario = (selectedRunSummary.scenario ?? "replay").replace(/[^a-z0-9_-]+/gi, "-");
      const link = document.createElement("a");
      link.href = source;
      link.download = `labverse-${scenario}-step-${selectedStep}-${replayMode}.png`;
      link.click();
    } catch (captureError) {
      setError(captureError instanceof Error ? captureError.message : "Could not save the replay scene.");
    } finally {
      setSavingScene(false);
    }
  }

  if (capture) {
    return (
      <div className="paper-capture-page">
        {capturedScene && !capture.autoplay ? (
          <img alt="Captured LabVerse replay scene" src={capturedScene} />
        ) : replayWorld ? (
          <div
            className={`paper-capture-stage${capture.autoplay ? " paper-capture-stage--video" : ""}`}
            aria-label="Preparing LabVerse replay capture"
          >
            <Suspense
              fallback={
                <div className="paper-capture-loading">
                  Loading replay renderer.
                </div>
              }
            >
              <PhaserGame
                ref={phaserGameRef}
                actionFeedback={replayFeedback}
                onSelectAgent={() => {}}
                onSelectEquipment={() => {}}
                onSelectRoom={() => {}}
                selectedAgentId={replayFrame?.actor_id ?? null}
                selectedEquipmentId={null}
                selectedRoomId={null}
                world={replayWorld}
              />
            </Suspense>
            {caption && (
              <div className="paper-capture-caption" aria-label="Current replay step">
                <strong>{caption.title}</strong>
                <span>{caption.body}</span>
              </div>
            )}
          </div>
        ) : (
          <div className="paper-capture-loading">
            {error ?? "Loading recorded LabVerse replay."}
          </div>
        )}
      </div>
    );
  }

  return (
    <div className="paper-studio">
      {/* ── Bundle bar ─────────────────────────────────────── */}
      <div className="paper-bundle-bar">
        <div className="paper-bundle-input">
          <span className="kicker">Bundle</span>
          <input
            onChange={(event) => setBundleDir(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === "Enter") void loadBundle(bundleDir);
            }}
            placeholder={DEFAULT_BUNDLE_DIR}
            spellCheck={false}
            autoCorrect="off"
            value={bundleDir}
          />
          <button
            className="btn-accent btn-sm"
            disabled={loadingBundle}
            onClick={() => void loadBundle(bundleDir)}
            type="button"
          >
            {loadingBundle ? "Loading…" : "Load"}
          </button>
        </div>
        <div className="paper-bundle-presets" aria-label="Included bundles">
          {BUNDLE_PRESETS.map((preset) => (
            <button
              className={loadedBundleDir === preset.path ? "active" : ""}
              key={preset.path}
              onClick={() => {
                setBundleDir(preset.path);
                void loadBundle(preset.path);
              }}
              type="button"
            >
              {preset.label}
            </button>
          ))}
        </div>

        {status ? (
          <div className="paper-bundle-meta">
            <div className="paper-bundle-meta-stat">
              <span>Completed</span>
              <strong>
                {status.completed_runs}
                <span className="json-sep"> / </span>
                {status.total_expected_runs}
              </strong>
            </div>
            <div className="paper-bundle-meta-stat">
              <span>Validation</span>
              <strong>
                <span
                  className={`chip chip-${status.validation_ok ? "ready" : "attention"}`}
                >
                  {status.validation_ok ? "validated" : "review needed"}
                </span>
              </strong>
            </div>
          </div>
        ) : (
          <div className="paper-bundle-meta">
            <div className="paper-bundle-meta-stat">
              <span>Status</span>
              <strong className="meta">no bundle loaded</strong>
            </div>
          </div>
        )}
      </div>

      {error && (
        <div className="error-banner">
          <strong>{error}</strong>
          <span>
            Expected local setup: <code>labverse paper-serve --host 0.0.0.0 --port 8001</code>
          </span>
        </div>
      )}

      <div className="paper-grid">
        {/* ── Rail ──────────────────────────────────────── */}
        <aside className="paper-rail">
          <div className="paper-section-head">
            <p className="kicker">Runs</p>
            <span className="paper-section-head-meta">
              {runs.length === 0 ? "—" : `${runs.length}`}
            </span>
          </div>

          {runs.length === 0 ? (
            <p className="paper-trace-empty">
              {loadingBundle ? "Loading bundle…" : "Load a bundle to list runs."}
            </p>
          ) : (
            <div className="paper-run-list" role="listbox" aria-label="Runs">
              {runs.map((run) => {
                const isActive = run.run_path === selectedRun;
                return (
                  <button
                    aria-selected={isActive}
                    className="paper-run-row"
                    key={run.run_path}
                    onClick={() => {
                      setSelectedRun(run.run_path);
                      setSelectedStep(null);
                      setReplayMode("after");
                    }}
                    role="option"
                    type="button"
                  >
                    <div className="paper-run-row-head">
                      <span className="paper-run-row-marker">{isActive ? "▸" : " "}</span>
                      <span className="paper-run-row-title">{shortRunLabel(run)}</span>
                      <span
                        className="paper-run-row-status"
                        data-tone={statusTone(run.status)}
                      >
                        {run.status ?? "unknown"}
                      </span>
                    </div>
                    <div className="paper-run-row-meta">
                      {run.condition && <span>{run.condition}</span>}
                      {run.model && <span>{run.model}</span>}
                      <span>{run.step_count} step{run.step_count === 1 ? "" : "s"}</span>
                      {run.replay_available ? (
                        <span>{run.replay_frame_count} frame{run.replay_frame_count === 1 ? "" : "s"}</span>
                      ) : (
                        <span>transcript only</span>
                      )}
                    </div>
                    {run.error && <div className="paper-run-row-error">{run.error}</div>}
                  </button>
                );
              })}
            </div>
          )}
        </aside>

        {/* ── Stage ─────────────────────────────────────── */}
        <section className="paper-stage">
          {selectedRunSummary ? (
            <header className="paper-stage-header">
              <p className="kicker paper-stage-kicker">Replay specimen</p>
              <h1 className="paper-stage-title">{shortRunLabel(selectedRunSummary)}</h1>
              <p className="paper-stage-subtitle">
                {selectedRunSummary.condition && <span>{selectedRunSummary.condition}</span>}
                {selectedRunSummary.model && <span>{selectedRunSummary.model}</span>}
                <span>{selectedRunSummary.step_count} step{selectedRunSummary.step_count === 1 ? "" : "s"}</span>
                <span>
                  {selectedRunSummary.replay_available
                    ? `${selectedRunSummary.replay_frame_count} replay frame${selectedRunSummary.replay_frame_count === 1 ? "" : "s"}`
                    : "transcript only"}
                </span>
              </p>
            </header>
          ) : (
            <header className="paper-stage-header">
              <p className="kicker paper-stage-kicker">Replay specimen</p>
              <h1 className="paper-stage-title">No run selected</h1>
              <p className="paper-stage-subtitle">
                <span>Choose a run from the rail to inspect its trace.</span>
              </p>
            </header>
          )}

          <div className="paper-stage-frame">
            {replayWorld ? (
              <Suspense
                fallback={
                  <div className="paper-replay-empty">
                    <div className="paper-replay-empty-inner">
                      <p className="kicker">Loading</p>
                      <p className="meta">Booting the replay engine.</p>
                    </div>
                  </div>
                }
              >
                <PhaserGame
                  ref={phaserGameRef}
                  actionFeedback={replayFeedback}
                  onSelectAgent={() => {}}
                  onSelectEquipment={() => {}}
                  onSelectRoom={() => {}}
                  selectedAgentId={replayFrame?.actor_id ?? null}
                  selectedEquipmentId={null}
                  selectedRoomId={null}
                  world={replayWorld}
                />
              </Suspense>
            ) : replayLoading ? (
              <div className="paper-replay-empty">
                <div className="paper-replay-empty-inner">
                  <p className="kicker">Loading</p>
                  <p className="meta">Fetching the recorded world snapshot for this step.</p>
                </div>
              </div>
            ) : (
              <div className="paper-replay-empty">
                <div className="paper-replay-empty-inner">
                  <p className="kicker">Replay unavailable</p>
                  <p className="meta">
                    {error
                      ? error
                      : stepDetailMatchesSelection && stepDetail?.compatibility_notice
                        ? stepDetail.compatibility_notice
                        : selectedStepSummary?.replay_available || selectedRunSummary?.replay_available
                          ? "Replay data is missing for the currently selected step."
                          : "Select a step to inspect the recorded trace."}
                  </p>
                </div>
              </div>
            )}
          </div>

          <div className="paper-stage-controls">
            <div className="paper-step-position">
              {selectedStep !== null && currentStepIndex >= 0 ? (
                <>
                  Step <strong>{currentStepIndex + 1}</strong>
                  <span>/</span>
                  <strong>{steps.length}</strong>
                </>
              ) : (
                <>—</>
              )}
            </div>

            <div className="paper-stage-buttons">
              <button
                className="btn-secondary"
                disabled={currentStepIndex <= 0}
                onClick={() => gotoOffset(-1)}
                type="button"
                title="Previous step (← / k)"
              >
                Prev<kbd>←</kbd>
              </button>
              <button
                className="btn-secondary"
                disabled={currentStepIndex < 0 || currentStepIndex >= steps.length - 1}
                onClick={() => gotoOffset(1)}
                type="button"
                title="Next step (→ / j)"
              >
                Next<kbd>→</kbd>
              </button>
              <button
                className="btn-secondary"
                disabled={!replayWorld || savingScene}
                onClick={() => void saveScene()}
                type="button"
                title="Save the current replay scene as a PNG"
              >
                {savingScene ? "Saving…" : "Save scene"}
              </button>
            </div>

            <div className="paper-mode-switch" role="tablist" aria-label="Replay world">
              <button
                className={replayMode === "before" ? "active" : ""}
                onClick={() => setReplayMode("before")}
                type="button"
                role="tab"
                aria-selected={replayMode === "before"}
                title="World before action (b)"
              >
                Before
              </button>
              <button
                className={replayMode === "after" ? "active" : ""}
                onClick={() => setReplayMode("after")}
                type="button"
                role="tab"
                aria-selected={replayMode === "after"}
                title="World after action (a)"
              >
                After
              </button>
            </div>
          </div>

          <div className="paper-filmstrip-wrap">
            <div className="paper-section-head">
              <p className="kicker">Filmstrip</p>
              <span className="paper-section-head-meta">
                {loadingSteps
                  ? "loading…"
                  : steps.length === 0
                    ? "no recorded steps"
                    : `${steps.length} step${steps.length === 1 ? "" : "s"}`}
              </span>
            </div>
            {steps.length > 0 && (
              <div className="paper-filmstrip" ref={filmstripRef} role="tablist">
                {steps.map((step) => {
                  const isActive = selectedStep === step.step_index;
                  return (
                    <button
                      aria-current={isActive}
                      className="paper-filmstrip-cell"
                      data-step-index={step.step_index}
                      data-tone={stepTone(step.status)}
                      key={step.step_index}
                      onClick={() => setSelectedStep(step.step_index)}
                      role="tab"
                      title={`Step ${step.step_index} · ${stepLabel(step)}`}
                      type="button"
                    >
                      <span className="paper-filmstrip-cell-num">
                        {String(step.step_index).padStart(2, "0")}
                      </span>
                      <span className="paper-filmstrip-cell-id">
                        {stepLabel(step)}
                      </span>
                      <span className="paper-filmstrip-cell-dot" aria-hidden="true" />
                    </button>
                  );
                })}
              </div>
            )}
          </div>

          <div className="paper-deltas">
            <p className="kicker">What changed</p>
            {loadingDetail ? (
              <p className="meta">Loading step delta…</p>
            ) : replayFrame?.change_summary.length ? (
              <ol>
                {replayFrame.change_summary.map((line, index) => (
                  <li key={`${line}-${index}`}>{line}</li>
                ))}
              </ol>
            ) : (
              <p className="meta">No state delta recorded for this step.</p>
            )}
          </div>
        </section>

        {/* ── Trace ─────────────────────────────────────── */}
        <aside className="paper-trace">
          {currentStep ? (
            <>
              <header className="paper-trace-head">
                <h2>Step {asString(currentStep.step_index) || selectedStep || "?"}</h2>
                <div className="paper-trace-meta">
                  <span>{asString(currentStep.actor_name ?? currentStep.actor_id) || "unknown actor"}</span>
                  <span>{asString(currentStep.prompt_surface_kind) || "no surface"}</span>
                  <span>{asString(currentStep.resolved_candidate_ref ?? currentStep.resolved_verb ?? currentStep.resolved_action_id) || "no action"}</span>
                  <span>{stepDetail?.replay_available ? "replay available" : "transcript only"}</span>
                </div>
              </header>

              <section className="paper-trace-section">
                <div className="paper-section-head">
                  <p className="kicker">System prompt</p>
                </div>
                {asString(currentStep.system_prompt) ? (
                  <pre className="paper-trace-text">{asString(currentStep.system_prompt)}</pre>
                ) : (
                  <p className="paper-trace-empty">no system prompt recorded</p>
                )}
              </section>

              <section className="paper-trace-section">
                <div className="paper-section-head">
                  <p className="kicker">User message</p>
                </div>
                {(() => {
                  const userMessage = currentStep.user_message as
                    | Record<string, unknown>
                    | undefined;
                  const content = asString(userMessage?.content);
                  if (content) {
                    return <pre className="paper-trace-text">{content}</pre>;
                  }
                  if (userMessage) {
                    return <JsonTree value={userMessage} maxAutoExpandDepth={2} />;
                  }
                  return <p className="paper-trace-empty">no user message recorded</p>;
                })()}
              </section>

              {attempts.length > 0 && (
                <section className="paper-trace-section">
                  <div className="paper-section-head">
                    <p className="kicker">Assistant attempts</p>
                    <span className="paper-section-head-meta">
                      {attempts.length} attempt{attempts.length === 1 ? "" : "s"}
                    </span>
                  </div>
                  {attempts.map((attempt) => (
                    <div className="paper-attempt" key={`attempt-${attempt.attempt_index ?? "x"}`}>
                      <div className="paper-attempt-head">
                        <span className="paper-attempt-label">
                          Attempt
                          <strong>{attempt.attempt_index ?? "?"}</strong>
                        </span>
                        <span className="paper-attempt-status">
                          {asString(attempt.metadata?.finish_reason) || "unknown"}
                        </span>
                      </div>
                      {attempt.response_tool_call ? (
                        <JsonTree
                          value={attempt.response_tool_call}
                          rootKey="tool_call"
                          maxAutoExpandDepth={3}
                        />
                      ) : attempt.response_text ? (
                        <pre className="paper-trace-text">{attempt.response_text}</pre>
                      ) : (
                        <p className="paper-trace-empty">no text or tool call</p>
                      )}
                      {attempt.invalid_reason && (
                        <div className="paper-attempt-error">invalid: {attempt.invalid_reason}</div>
                      )}
                      {attempt.provider_error && (
                        <div className="paper-attempt-error">provider: {attempt.provider_error}</div>
                      )}
                    </div>
                  ))}
                </section>
              )}

              {currentStep.action_result !== undefined && currentStep.action_result !== null && (
                <section className="paper-trace-section">
                  <div className="paper-section-head">
                    <p className="kicker">Action result</p>
                  </div>
                  <JsonTree
                    value={currentStep.action_result}
                    rootKey="action_result"
                    maxAutoExpandDepth={2}
                  />
                </section>
              )}

              {currentStep.tool_schema !== undefined && currentStep.tool_schema !== null && (
                <section className="paper-trace-section">
                  <div className="paper-section-head">
                    <p className="kicker">Tool schema</p>
                  </div>
                  <JsonTree value={currentStep.tool_schema} maxAutoExpandDepth={1} />
                </section>
              )}

              <section className="paper-trace-section">
                <div className="paper-section-head">
                  <p className="kicker">Raw step</p>
                </div>
                <JsonTree value={currentStep} maxAutoExpandDepth={1} />
              </section>
            </>
          ) : (
            <>
              <header className="paper-trace-head">
                <h2>Trace</h2>
                <div className="paper-trace-meta">
                  <span>no step selected</span>
                </div>
              </header>
              <p className="paper-trace-empty">
                {loadingDetail
                  ? "loading step detail…"
                  : "Choose a run and step to inspect the prompt, attempts, and replay frame."}
              </p>
            </>
          )}
        </aside>
      </div>
    </div>
  );
}

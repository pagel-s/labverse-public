export interface Position {
  x: number;
  y: number;
}

export interface Zone {
  id: string;
  name: string;
  position: Position;
  width: number;
  height: number;
  zone_type: string;
  capacity: number;
}

export interface Room {
  id: string;
  name: string;
  room_type: string;
  bsl_level: number;
  temperature_celsius: number;
  humidity_percent: number;
  has_ventilation: boolean;
  has_emergency_shower: boolean;
  has_eyewash: boolean;
  max_occupancy: number;
  position: Position;
  width: number;
  height: number;
  zones: Zone[];
  equipment_ids: string[];
  material_ids: string[];
}

export interface Door {
  id: string;
  connects: [string, string];
  is_open: boolean;
  locked: boolean;
  requires_keycard: boolean;
  requires_bsl_clearance: number | null;
}

export interface LabLayout {
  id: string;
  name: string;
  description: string;
  rooms: Record<string, Room>;
  doors: Door[];
  default_entry_room: string;
}

export interface EquipmentCommandParameter {
  name: string;
  description: string;
  param_type: string;
  unit: string | null;
  min_value: number | null;
  max_value: number | null;
  allowed_values: string[] | null;
  default_value: unknown;
  current_value: unknown;
}

export interface EquipmentCommand {
  name: string;
  description: string;
  parameters: EquipmentCommandParameter[];
  duration_seconds: number | null;
  preconditions: string[];
}

export interface EquipmentParameter {
  name: string;
  description: string;
  param_type: string;
  unit: string | null;
  min_value: number | null;
  max_value: number | null;
  allowed_values: string[] | null;
  default_value: unknown;
  current_value: unknown;
}

export interface Equipment {
  id: string;
  name: string;
  equipment_type: string;
  manufacturer: string | null;
  model_name: string | null;
  status: string;
  room_id: string;
  zone_id: string | null;
  position: Position | null;
  commands: Record<string, EquipmentCommand>;
  running_command: string | null;
  running_command_started_at: string | null;
  running_command_duration: number | null;
  error_message: string | null;
  properties: Record<string, EquipmentParameter>;
  required_certifications: string[];
  max_concurrent_users: number;
  current_users: string[];
  last_maintenance: string | null;
  maintenance_interval_days: number | null;
  metadata: Record<string, string>;
}

export interface GhsProfile {
  pictograms: string[];
  signal_word: string;
  h_statements: string[];
  p_statements: string[];
  storage_codes: string[];
}

export interface Material {
  id: string;
  name: string;
  cas_number: string | null;
  formula: string | null;
  room_id: string;
  zone_id: string | null;
  phase: string;
  ghs: GhsProfile;
  concentration: string | null;
  quantity: number;
  quantity_unit: string;
  storage_temperature_celsius: number | null;
  expiry_date: string | null;
  lot_number: string | null;
  supplier: string | null;
  incompatible_with: string[];
  required_ppe: string[];
  waste_category: string;
  is_controlled: boolean;
}

export interface Sample {
  id: string;
  name: string;
  sample_type: string;
  source: string | null;
  container_type: string | null;
  volume_ul: number | null;
  concentration: string | null;
  room_id: string;
  zone_id: string | null;
  equipment_id: string | null;
  status: string;
  bsl_level: number | null;
  created_at: string | null;
  created_by: string | null;
  derived_from: string[];
  metadata: Record<string, string>;
}

export interface Certification {
  cert_type: string;
  granted_date: string;
  expiry_date: string | null;
  granted_by: string | null;
}

export interface Person {
  id: string;
  name: string;
  role: string;
  certifications: Certification[];
  current_room_id: string | null;
  current_zone_id: string | null;
  current_ppe: string[];
  carrying: string[];
  lab_group: string | null;
  current_activity: string | null;
  is_active: boolean;
}

export interface PersonSummary {
  id: string;
  name: string;
  role: string;
  current_zone_id: string | null;
  current_activity: string | null;
}

export interface EquipmentSummary {
  id: string;
  name: string;
  equipment_type: string;
  status: string;
  zone_id: string | null;
  running_command: string | null;
}

export interface MaterialSummary {
  id: string;
  name: string;
  quantity: number;
  quantity_unit: string;
  zone_id: string | null;
  is_hazardous: boolean;
}

export interface SampleSummary {
  id: string;
  name: string;
  sample_type: string;
  status: string;
  zone_id: string | null;
  equipment_id: string | null;
  bsl_level: number | null;
  volume_ul: number | null;
}

export interface DoorSummary {
  id: string;
  connects: [string, string];
  is_open: boolean;
  locked: boolean;
  requires_keycard: boolean;
  requires_bsl_clearance: number | null;
}

export interface SelfState {
  id: string;
  name: string;
  role: string;
  current_room_id: string | null;
  current_zone_id: string | null;
  current_ppe: string[];
  carrying: string[];
  current_activity: string | null;
  certifications: string[];
}

export interface Observation {
  observation_id: string;
  agent_id: string;
  timestamp: string;
  tick: number;
  current_room: Room;
  self_state: SelfState;
  visible_persons: PersonSummary[];
  visible_equipment: EquipmentSummary[];
  visible_materials: MaterialSummary[];
  visible_samples: SampleSummary[];
  visible_doors: DoorSummary[];
  carrying: string[];
  current_ppe: string[];
  active_alerts: string[];
  recent_events: string[];
  working_memory: string;
  trace_parent_action_id: string | null;
  narrative: string;
}

export interface AvailableAction {
  candidate_ref: string;
  group_key: string;
  group_label: string;
  actor_id: string | null;
  label: string;
  verb: string;
  category: string;
  description: string;
  target_id: string | null;
  target_type: string | null;
  target_label: string | null;
  fixed_parameters: Record<string, unknown>;
  required_parameters: string[];
  optional_parameters: string[];
  parameter_details: Record<string, string>;
  free_parameter_schema: Record<string, unknown>;
  estimated_duration_seconds: number | null;
}

export interface ActionGroup {
  group_key: string;
  label: string;
  candidates: AvailableAction[];
}

export interface ActionSpec {
  agent_id: string | null;
  tick: number | null;
  groups: ActionGroup[];
  context: string;
}

export interface ChangedEntity {
  entity_type: string;
  entity_id: string;
  summary: string;
  room_id: string | null;
  zone_id: string | null;
}

export interface ActionResult {
  action_id: string;
  success: boolean;
  status: "succeeded" | "blocked" | "failed" | null;
  outcome_description: string;
  state_changes: Record<string, unknown>[];
  changed_entities: ChangedEntity[];
  time_elapsed_seconds: number;
  safety_violations: string[];
  warnings: string[];
  blocked_reason: string | null;
  failure_reason: string | null;
  emitted_event_ids: string[];
  post_observation_id: string | null;
}

export interface AgentTraceRecord {
  trace_id: string;
  agent_id: string;
  tick: number;
  timestamp: string;
  observation: Observation;
  action_spec: ActionSpec;
  action: {
    id: string;
    agent_id: string;
    category: string;
    verb: string;
    target_id: string | null;
    parameters: Record<string, unknown>;
    timestamp: string | null;
    protocol_step_id: string | null;
  };
  result: ActionResult;
  event_ids: string[];
  post_observation: Observation | null;
}

export interface AgentMemorySnapshot {
  agent_id: string;
  working_memory: string;
  last_observation_id: string | null;
  last_trace_id: string | null;
  last_action_status: "succeeded" | "blocked" | "failed" | null;
  trace_count: number;
  recent_trace_ids: string[];
}

export interface WorldState {
  lab: LabLayout;
  equipment: Record<string, Equipment>;
  materials: Record<string, Material>;
  samples: Record<string, Sample>;
  persons: Record<string, Person>;
}

export interface SimulationStatus {
  loaded: boolean;
  config_path: string | null;
  is_playing: boolean;
  tick: number;
  current_time: string | null;
  entity_counts: Record<string, number>;
}

export interface SimulationEvent {
  event_type: string;
  tick: number;
  timestamp: string;
  source_id: string | null;
  target_id: string | null;
  description: string;
  data: Record<string, unknown>;
}

export interface QueryResponse {
  answer: string;
}

export interface PaperBundleStatus {
  bundle_dir: string;
  total_expected_runs: number;
  completed_runs: number;
  running_runs: string[];
  latest_run: string | null;
  validation_ok: boolean;
  validation_issues: string[];
  diagnostics_clear: boolean;
  blocking_reasons: string[];
}

export interface PaperRunSummary {
  run_path: string;
  condition: string | null;
  model: string | null;
  scenario: string | null;
  replication: number | null;
  status: string | null;
  step_count: number;
  replay_available: boolean;
  replay_frame_count: number;
  error: string | null;
}

export interface PaperChatViewResponse {
  run_path: string | null;
  recorded_steps: number;
  transcript: string;
}

export interface PaperStepSummary {
  run_path: string;
  step_index: number;
  actor_id: string | null;
  actor_name: string | null;
  prompt_surface_kind: string | null;
  status: string | null;
  resolved_candidate_ref: string | null;
  resolved_action_id: string | null;
  resolved_verb: string | null;
  resolved_target_id: string | null;
  action_status: string | null;
  error: string | null;
  replay_available: boolean;
}

export interface PaperReplayFrame {
  step_index: number;
  actor_id: string | null;
  resolved_candidate_ref: string | null;
  resolved_action_id: string | null;
  resolved_verb: string | null;
  resolved_target_id: string | null;
  resolved_action_parameters: Record<string, unknown>;
  action_result: Record<string, unknown> | null;
  world_before: WorldState;
  world_after: WorldState;
  tick_before: number;
  tick_after: number;
  current_time_before: string | null;
  current_time_after: string | null;
  change_summary: string[];
  trace_ids: string[];
  event_ids: string[];
}

export interface PaperStepDetail {
  run_path: string;
  recorded_steps: number;
  step_index: number;
  step: Record<string, unknown>;
  replay_available: boolean;
  compatibility_notice: string | null;
  replay_frame: PaperReplayFrame | null;
}

const API_BASE = import.meta.env.VITE_LABVERSE_API_BASE_URL ?? "/api";

const REQUEST_TIMEOUT_MS = 30_000;
const PAPER_API_HELP =
  "Paper API is not reachable. Start `labverse paper-serve --host 0.0.0.0 --port 8001`, keep the Vite dev server on 5173, then reload.";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);

  let response: Response;
  try {
    response = await fetch(`${API_BASE}${path}`, {
      headers: {
        "Content-Type": "application/json",
        ...(init?.headers ?? {}),
      },
      ...init,
      signal: init?.signal ?? controller.signal,
    });
  } catch (err) {
    if (err instanceof DOMException && err.name === "AbortError") {
      throw new Error(`Request to ${path} timed out after ${REQUEST_TIMEOUT_MS / 1000}s`);
    }
    if (err instanceof TypeError) {
      throw new Error(PAPER_API_HELP);
    }
    throw err;
  } finally {
    clearTimeout(timeout);
  }

  if (!response.ok) {
    let message = `${response.status} ${response.statusText}`;
    try {
      const payload = (await response.json()) as { detail?: string };
      if (payload.detail) {
        message = payload.detail;
      }
    } catch {
      // Keep the HTTP status fallback.
    }
    if ((response.status === 404 || response.status === 500 || response.status === 502) && API_BASE === "/api") {
      message = PAPER_API_HELP;
    }
    throw new Error(message);
  }

  return (await response.json()) as T;
}

function withQuery(path: string, query: Record<string, string | boolean | undefined>): string {
  const params = new URLSearchParams();
  for (const [key, value] of Object.entries(query)) {
    if (value !== undefined) {
      params.set(key, String(value));
    }
  }
  const queryString = params.toString();
  return queryString ? `${path}?${queryString}` : path;
}

export const apiClient = {
  getPaperHealth(): Promise<{ status: string; service: string }> {
    return request<{ status: string; service: string }>("/paper/health");
  },

  getPaperBundleStatus(bundleDir: string): Promise<PaperBundleStatus> {
    return request<PaperBundleStatus>(
      withQuery("/paper/status", { bundle_dir: bundleDir }),
    );
  },

  listPaperRuns(bundleDir: string): Promise<PaperRunSummary[]> {
    return request<PaperRunSummary[]>(
      withQuery("/paper/runs", { bundle_dir: bundleDir }),
    );
  },

  listPaperSteps(bundleDir: string, run?: string | null): Promise<PaperStepSummary[]> {
    return request<PaperStepSummary[]>(
      withQuery("/paper/steps", {
        bundle_dir: bundleDir,
        run: run ?? undefined,
      }),
    );
  },

  getPaperStepDetail(options: {
    bundleDir: string;
    run: string;
    step: number;
  }): Promise<PaperStepDetail> {
    return request<PaperStepDetail>(
      withQuery("/paper/step", {
        bundle_dir: options.bundleDir,
        run: options.run,
        step: String(options.step),
      }),
    );
  },

  getPaperChatView(options: {
    bundleDir: string;
    run?: string | null;
    step?: number | null;
    tail?: number;
    showToolSchema?: boolean;
  }): Promise<PaperChatViewResponse> {
    return request<PaperChatViewResponse>(
      withQuery("/paper/chat-view", {
        bundle_dir: options.bundleDir,
        run: options.run ?? undefined,
        step: options.step === null || options.step === undefined ? undefined : String(options.step),
        tail: String(options.tail ?? 1),
        show_tool_schema: options.showToolSchema ?? false,
      }),
    );
  },
};

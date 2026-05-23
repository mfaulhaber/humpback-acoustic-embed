import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";

export type ContinuousEmbeddingSourceKind = "surfperch" | "region_crnn";

export interface ContinuousEmbeddingJob {
  id: string;
  status: string;
  event_segmentation_job_id: string | null;
  event_source_mode: "raw" | "effective";
  model_version: string;
  window_size_seconds: number | null;
  hop_seconds: number | null;
  pad_seconds: number | null;
  target_sample_rate: number;
  feature_config_json: string | null;
  encoding_signature: string;
  vector_dim: number | null;
  total_events: number | null;
  merged_spans: number | null;
  total_windows: number | null;
  parquet_path: string | null;
  error_message: string | null;
  region_detection_job_id: string | null;
  chunk_size_seconds: number | null;
  chunk_hop_seconds: number | null;
  crnn_checkpoint_sha256: string | null;
  crnn_segmentation_model_id: string | null;
  projection_kind: string | null;
  projection_dim: number | null;
  total_regions: number | null;
  total_chunks: number | null;
  created_at: string;
  updated_at: string;
}

export interface ContinuousEmbeddingSpanSummary {
  merged_span_id: number;
  event_id: string;
  region_id: string;
  start_timestamp: number;
  end_timestamp: number;
  window_count: number;
}

export interface ContinuousEmbeddingRegionSummary {
  region_id: string;
  start_timestamp: number;
  end_timestamp: number;
  chunk_count: number;
}

export interface ContinuousEmbeddingJobManifest {
  job_id: string;
  model_version: string;
  source_kind: ContinuousEmbeddingSourceKind;
  event_source_mode?: "raw" | "effective";
  vector_dim: number;
  target_sample_rate: number;
  window_size_seconds?: number | null;
  hop_seconds?: number | null;
  pad_seconds?: number | null;
  total_events?: number | null;
  merged_spans?: number | null;
  total_windows?: number | null;
  spans?: ContinuousEmbeddingSpanSummary[];
  region_detection_job_id?: string | null;
  event_segmentation_job_id?: string | null;
  crnn_checkpoint_sha256?: string | null;
  chunk_size_seconds?: number | null;
  chunk_hop_seconds?: number | null;
  projection_kind?: string | null;
  projection_dim?: number | null;
  total_regions?: number | null;
  total_chunks?: number | null;
  regions?: ContinuousEmbeddingRegionSummary[];
}

export interface ContinuousEmbeddingJobDetail {
  job: ContinuousEmbeddingJob;
  manifest: ContinuousEmbeddingJobManifest | null;
  extra?: Record<string, unknown> | null;
}

export interface EventEncoderPoolingConfig {
  enabled_pools?: (
    | "mean_pool"
    | "top_k_pool"
    | "start_pool"
    | "middle_pool"
    | "end_pool"
  )[];
  top_k_fraction?: number;
  min_overlap_fraction?: number;
  min_chunks_per_event?: number;
}

export interface EventEncoderDescriptorConfig {
  target_sample_rate?: number;
  n_fft?: number;
  hop_length?: number;
  eps?: number;
  ridge_min_frequency_hz?: number;
  ridge_max_frequency_hz?: number;
  ridge_candidate_count?: number;
  ridge_smoothness_penalty?: number;
  ridge_peak_prominence_ratio?: number;
  ridge_summary_low_percentile?: number;
  ridge_summary_high_percentile?: number;
  band_peak_min_frequency_hz?: number;
  band_peak_max_frequency_hz?: number | null;
  high_band_min_frequency_hz?: number;
}

export interface EventEncoderPreprocessingConfig {
  l2_normalize_pools?: boolean;
  pca_dim?: 64 | 128;
  embedding_weight?: number;
  descriptor_weight?: number;
  descriptor_clip_value?: number | null;
}

export interface EventEncoderJob {
  id: string;
  status: string;
  event_segmentation_job_id: string;
  event_source_mode: "raw" | "effective";
  continuous_embedding_job_id: string;
  continuous_embedding_signature: string;
  tokenizer_version: string;
  pooling_config_json: string;
  descriptor_config_json: string;
  preprocessing_config_json: string;
  k_values_json: string;
  random_seed: number;
  tokenization_signature: string;
  event_vector_dim: number | null;
  total_events: number | null;
  encoded_events: number | null;
  skipped_events: number | null;
  event_vectors_path: string | null;
  event_tokens_path: string | null;
  token_sequences_path: string | null;
  manifest_path: string | null;
  report_path: string | null;
  error_message: string | null;
  created_at: string;
  updated_at: string;
}

export interface EventEncoderJobDetail {
  job: EventEncoderJob;
  manifest: Record<string, unknown> | null;
  report: Record<string, unknown> | null;
}

export interface EventEncoderTimelineEvent {
  event_id: string;
  region_id: string;
  source_sequence_key: string;
  sequence_index: number;
  start_timestamp: number;
  end_timestamp: number;
  token_id: number;
  token_label: string;
  token_confidence: number;
  distance_to_centroid: number;
  second_centroid_distance: number | null;
  descriptor_values: Record<string, number>;
  descriptor_vector_values: Record<string, number>;
}

export type PianoRollNotesJobStatus =
  | "queued"
  | "running"
  | "complete"
  | "failed"
  | "canceled";

export interface PianoRollNotesJobRead {
  id: string;
  event_encoder_job_id: string;
  extractor_version: string;
  status: PianoRollNotesJobStatus;
  started_at: string | null;
  finished_at: string | null;
  error_message: string | null;
  notes_path: string | null;
  n_events: number | null;
  n_notes: number | null;
  compute_seconds: number | null;
  params_json: string;
  created_at: string;
  updated_at: string;
}

export interface PianoRollNotesStatusAbsent {
  status: "absent";
}

export type PianoRollNotesStatus =
  | PianoRollNotesJobRead
  | PianoRollNotesStatusAbsent;

export function isPianoRollNotesStatusAbsent(
  status: PianoRollNotesStatus,
): status is PianoRollNotesStatusAbsent {
  return status.status === "absent";
}

export interface PianoRollNote {
  event_id: string;
  event_token: number;
  partial_index: number;
  midi_pitch: number;
  start_utc: number;
  start_offset_s: number;
  duration_s: number;
  velocity: number;
  peak_magnitude: number;
  track_id: number;
  note_uid: string | null;
  f0_track_id: number | null;
  contour_frame_count: number | null;
}

export interface PianoRollNotesResponse {
  job_id: string;
  extractor_version: string;
  n_notes: number;
  notes: PianoRollNote[];
}

export interface PianoRollNoteContourFrame {
  frame_index: number;
  time_offset_s: number;
  cents_from_pitch: number;
  harmonic_strength: number;
  subharmonic_octave: number;
}

export interface PianoRollNoteContourResponse {
  job_id: string;
  extractor_version: string;
  n_notes: number;
  contours: Record<string, PianoRollNoteContourFrame[]>;
}

export const PIANO_ROLL_NOTE_CONTOUR_BATCH_LIMIT = 2000;

export interface CreatePianoRollNotesJobRequest {
  extractor_version?: string;
  params?: Record<string, unknown>;
}

export type PianoRollMidiExportJobStatus =
  | "queued"
  | "running"
  | "complete"
  | "failed"
  | "canceled";

export interface PianoRollMidiExportRead {
  id: string;
  event_encoder_job_id: string;
  extractor_version: string;
  status: PianoRollMidiExportJobStatus;
  started_at: string | null;
  finished_at: string | null;
  error_message: string | null;
  midi_path: string | null;
  n_notes: number | null;
  n_bytes: number | null;
  compute_seconds: number | null;
  params_json: string;
  window_start_utc: number;
  window_end_utc: number;
  audio_path: string;
  audio_size_bytes: number;
  audio_sample_rate: number;
  audio_duration_s: number;
  created_at: string;
  updated_at: string;
}

export const MAX_EXPORT_WINDOW_SECONDS = 1800;
// Server-side cache-hit tolerance is 1 ms (see services/piano_roll_midi_export_service.py);
// the frontend uses a looser ~50 ms tolerance only to decide whether to emphasize
// the "Re-export view" button.

export interface PianoRollMidiExportStatusAbsent {
  status: "absent";
}

export type PianoRollMidiExportStatus =
  | PianoRollMidiExportRead
  | PianoRollMidiExportStatusAbsent;

export function isPianoRollMidiExportStatusAbsent(
  status: PianoRollMidiExportStatus,
): status is PianoRollMidiExportStatusAbsent {
  return status.status === "absent";
}

export interface CreatePianoRollMidiExportRequest {
  extractor_version?: string;
  params?: Record<string, unknown>;
  force?: boolean;
  window_start_utc: number;
  window_end_utc: number;
}

export interface EventEncoderTimelineResponse {
  job_id: string;
  event_segmentation_job_id: string;
  event_source_mode: "raw" | "effective";
  continuous_embedding_job_id: string;
  region_detection_job_id: string;
  selected_k: number;
  valid_k_values: number[];
  descriptor_feature_names: string[];
  descriptor_feature_units: Record<string, string>;
  job_start_timestamp: number;
  job_end_timestamp: number;
  events: EventEncoderTimelineEvent[];
  notes_status: PianoRollNotesStatus;
}

export type EventEncoderProjectionMethod = "umap" | "pca";

export interface EventEncoderProjectionPoint {
  event_id: string;
  region_id: string;
  source_sequence_key: string;
  sequence_index: number;
  start_timestamp: number;
  end_timestamp: number;
  token_id: number;
  token_label: string;
  token_confidence: number;
  distance_to_centroid: number;
  second_centroid_distance: number | null;
  x: number;
  y: number;
}

export interface EventEncoderProjectionResponse {
  job_id: string;
  selected_k: number;
  valid_k_values: number[];
  method: EventEncoderProjectionMethod;
  x_axis_label: string;
  y_axis_label: string;
  points: EventEncoderProjectionPoint[];
}

export interface CreateContinuousEmbeddingJobRequest {
  event_segmentation_job_id?: string;
  event_source_mode?: "raw" | "effective";
  model_version?: string;
  hop_seconds?: number;
  pad_seconds?: number;
  region_detection_job_id?: string;
  crnn_segmentation_model_id?: string;
  chunk_size_seconds?: number;
  chunk_hop_seconds?: number;
  projection_kind?: "identity" | "random" | "pca";
  projection_dim?: number;
}

export interface CreateEventEncoderJobRequest {
  event_segmentation_job_id: string;
  event_source_mode?: "raw" | "effective";
  continuous_embedding_job_id: string;
  tokenizer_version?: string;
  pooling?: EventEncoderPoolingConfig;
  descriptor?: EventEncoderDescriptorConfig;
  preprocessing?: EventEncoderPreprocessingConfig;
  k_values?: number[];
  random_seed?: number;
}

export function continuousEmbeddingSourceKind(
  job: Pick<ContinuousEmbeddingJob, "model_version" | "region_detection_job_id">,
): ContinuousEmbeddingSourceKind {
  if (job.region_detection_job_id) return "region_crnn";
  if (job.model_version.startsWith("crnn-")) return "region_crnn";
  return "surfperch";
}

const ROOT = "/sequence-models/continuous-embeddings";
const EVENT_ENCODER_ROOT = "/sequence-models/event-encoders";

export class ApiError extends Error {
  constructor(
    public status: number,
    message: string,
  ) {
    super(message);
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(path, init);
  if (!res.ok) {
    const text = await res.text().catch(() => res.statusText);
    throw new ApiError(res.status, text);
  }
  if (res.status === 204) return undefined as T;
  return res.json();
}

export function fetchContinuousEmbeddingJobs(
  status?: string,
): Promise<ContinuousEmbeddingJob[]> {
  const q = status ? `?status=${encodeURIComponent(status)}` : "";
  return request<ContinuousEmbeddingJob[]>(`${ROOT}${q}`);
}

export function fetchContinuousEmbeddingJob(
  jobId: string,
): Promise<ContinuousEmbeddingJobDetail> {
  return request<ContinuousEmbeddingJobDetail>(`${ROOT}/${jobId}`);
}

export function createContinuousEmbeddingJob(
  body: CreateContinuousEmbeddingJobRequest,
): Promise<ContinuousEmbeddingJob> {
  return request<ContinuousEmbeddingJob>(ROOT, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}

export function cancelContinuousEmbeddingJob(
  jobId: string,
): Promise<ContinuousEmbeddingJob> {
  return request<ContinuousEmbeddingJob>(`${ROOT}/${jobId}/cancel`, {
    method: "POST",
  });
}

export function deleteContinuousEmbeddingJob(jobId: string): Promise<void> {
  return request<void>(`${ROOT}/${jobId}`, { method: "DELETE" });
}

export function fetchEventEncoderJobs(
  status?: string,
): Promise<EventEncoderJob[]> {
  const q = status ? `?status=${encodeURIComponent(status)}` : "";
  return request<EventEncoderJob[]>(`${EVENT_ENCODER_ROOT}${q}`);
}

export function fetchEventEncoderJob(
  jobId: string,
): Promise<EventEncoderJobDetail> {
  return request<EventEncoderJobDetail>(`${EVENT_ENCODER_ROOT}/${jobId}`);
}

export function fetchEventEncoderTimeline(
  jobId: string,
  k?: number | null,
): Promise<EventEncoderTimelineResponse> {
  const q = k == null ? "" : `?k=${encodeURIComponent(String(k))}`;
  return request<EventEncoderTimelineResponse>(
    `${EVENT_ENCODER_ROOT}/${jobId}/timeline${q}`,
  );
}

export function fetchEventEncoderProjection(
  jobId: string,
  method: EventEncoderProjectionMethod,
  k?: number | null,
): Promise<EventEncoderProjectionResponse> {
  const params = new URLSearchParams({ method });
  if (k != null) params.set("k", String(k));
  return request<EventEncoderProjectionResponse>(
    `${EVENT_ENCODER_ROOT}/${jobId}/projection?${params.toString()}`,
  );
}

export function createEventEncoderJob(
  body: CreateEventEncoderJobRequest,
): Promise<EventEncoderJob> {
  return request<EventEncoderJob>(EVENT_ENCODER_ROOT, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}

export function cancelEventEncoderJob(jobId: string): Promise<EventEncoderJob> {
  return request<EventEncoderJob>(`${EVENT_ENCODER_ROOT}/${jobId}/cancel`, {
    method: "POST",
  });
}

export function deleteEventEncoderJob(jobId: string): Promise<void> {
  return request<void>(`${EVENT_ENCODER_ROOT}/${jobId}`, { method: "DELETE" });
}

export function fetchPianoRollNotesStatus(
  jobId: string,
): Promise<PianoRollNotesStatus> {
  return request<PianoRollNotesStatus>(
    `${EVENT_ENCODER_ROOT}/${jobId}/notes-status`,
  );
}

export interface PianoRollNotesViewport {
  startUtc?: number | null;
  endUtc?: number | null;
  eventIds?: string[] | null;
  extractorVersion?: string | null;
}

export interface PianoRollNoteContourQuery {
  noteUids: string[];
  extractorVersion?: string | null;
}

export function fetchPianoRollNoteContours(
  jobId: string,
  query: PianoRollNoteContourQuery,
): Promise<PianoRollNoteContourResponse> {
  return request<PianoRollNoteContourResponse>(
    `${EVENT_ENCODER_ROOT}/${jobId}/notes/contours`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        note_uids: query.noteUids,
        extractor_version: query.extractorVersion ?? null,
      }),
    },
  );
}

export function fetchPianoRollNotes(
  jobId: string,
  viewport: PianoRollNotesViewport = {},
): Promise<PianoRollNotesResponse> {
  const params = new URLSearchParams();
  if (viewport.startUtc != null) params.set("start_utc", String(viewport.startUtc));
  if (viewport.endUtc != null) params.set("end_utc", String(viewport.endUtc));
  if (viewport.eventIds) {
    for (const id of viewport.eventIds) {
      params.append("event_ids", id);
    }
  }
  if (viewport.extractorVersion) {
    params.set("extractor_version", viewport.extractorVersion);
  }
  const qs = params.toString();
  return request<PianoRollNotesResponse>(
    `${EVENT_ENCODER_ROOT}/${jobId}/notes${qs ? `?${qs}` : ""}`,
  );
}

export function createPianoRollNotesJob(
  jobId: string,
  body: CreatePianoRollNotesJobRequest = {},
): Promise<PianoRollNotesJobRead> {
  return request<PianoRollNotesJobRead>(
    `${EVENT_ENCODER_ROOT}/${jobId}/notes-jobs`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    },
  );
}

export function fetchPianoRollMidiExportStatus(
  jobId: string,
): Promise<PianoRollMidiExportStatus> {
  return request<PianoRollMidiExportStatus>(
    `${EVENT_ENCODER_ROOT}/${jobId}/midi-export-status`,
  );
}

export function createPianoRollMidiExport(
  jobId: string,
  body: CreatePianoRollMidiExportRequest,
): Promise<PianoRollMidiExportRead> {
  return request<PianoRollMidiExportRead>(
    `${EVENT_ENCODER_ROOT}/${jobId}/midi-exports`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    },
  );
}

export function pianoRollMidiExportDownloadUrl(jobId: string): string {
  return `${EVENT_ENCODER_ROOT}/${jobId}/midi-export`;
}

export function pianoRollAudioExportDownloadUrl(jobId: string): string {
  return `${EVENT_ENCODER_ROOT}/${jobId}/audio-export`;
}

const ACTIVE_STATUSES = new Set(["queued", "running"]);

export function isContinuousEmbeddingJobActive(
  job: Pick<ContinuousEmbeddingJob, "status">,
): boolean {
  return ACTIVE_STATUSES.has(job.status);
}

export function isEventEncoderJobActive(
  job: Pick<EventEncoderJob, "status">,
): boolean {
  return ACTIVE_STATUSES.has(job.status);
}

export function useContinuousEmbeddingJobs(refetchInterval = 3000) {
  return useQuery({
    queryKey: ["continuous-embedding-jobs"],
    queryFn: () => fetchContinuousEmbeddingJobs(),
    refetchInterval,
  });
}

export function useContinuousEmbeddingJob(jobId: string | null) {
  return useQuery({
    queryKey: ["continuous-embedding-job", jobId],
    queryFn: () => fetchContinuousEmbeddingJob(jobId as string),
    enabled: jobId != null,
    refetchInterval: (query) => {
      const data = query.state.data as ContinuousEmbeddingJobDetail | undefined;
      if (!data) return 3000;
      return isContinuousEmbeddingJobActive(data.job) ? 3000 : false;
    },
  });
}

export function useEventEncoderJobs(refetchInterval = 3000) {
  return useQuery({
    queryKey: ["event-encoder-jobs"],
    queryFn: () => fetchEventEncoderJobs(),
    refetchInterval,
  });
}

export function useEventEncoderJob(jobId: string | null) {
  return useQuery({
    queryKey: ["event-encoder-job", jobId],
    queryFn: () => fetchEventEncoderJob(jobId as string),
    enabled: jobId != null,
    refetchInterval: (query) => {
      const data = query.state.data as EventEncoderJobDetail | undefined;
      if (!data) return 3000;
      return isEventEncoderJobActive(data.job) ? 3000 : false;
    },
  });
}

export function useEventEncoderTimeline(
  jobId: string | null,
  k: number | null,
  enabled: boolean,
) {
  return useQuery({
    queryKey: ["event-encoder-timeline", jobId, k],
    queryFn: () => fetchEventEncoderTimeline(jobId as string, k),
    enabled: enabled && jobId != null,
  });
}

export function useEventEncoderProjection(
  jobId: string | null,
  k: number | null,
  method: EventEncoderProjectionMethod,
  enabled: boolean,
) {
  return useQuery({
    queryKey: ["event-encoder-projection", jobId, k, method],
    queryFn: () => fetchEventEncoderProjection(jobId as string, method, k),
    enabled: enabled && jobId != null,
  });
}

export function useCreateContinuousEmbeddingJob() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (body: CreateContinuousEmbeddingJobRequest) =>
      createContinuousEmbeddingJob(body),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["continuous-embedding-jobs"] });
    },
  });
}

export function useCreateEventEncoderJob() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (body: CreateEventEncoderJobRequest) =>
      createEventEncoderJob(body),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["event-encoder-jobs"] });
    },
  });
}

export function useCancelContinuousEmbeddingJob() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (jobId: string) => cancelContinuousEmbeddingJob(jobId),
    onSuccess: (_data, jobId) => {
      qc.invalidateQueries({ queryKey: ["continuous-embedding-jobs"] });
      qc.invalidateQueries({ queryKey: ["continuous-embedding-job", jobId] });
    },
  });
}

export function useDeleteContinuousEmbeddingJob() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (jobId: string) => deleteContinuousEmbeddingJob(jobId),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["continuous-embedding-jobs"] });
    },
  });
}

export function useCancelEventEncoderJob() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (jobId: string) => cancelEventEncoderJob(jobId),
    onSuccess: (_data, jobId) => {
      qc.invalidateQueries({ queryKey: ["event-encoder-jobs"] });
      qc.invalidateQueries({ queryKey: ["event-encoder-job", jobId] });
    },
  });
}

export function useDeleteEventEncoderJob() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (jobId: string) => deleteEventEncoderJob(jobId),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["event-encoder-jobs"] });
    },
  });
}

const ACTIVE_NOTES_STATUSES = new Set<PianoRollNotesJobStatus>(["queued", "running"]);

function isActiveNotesStatus(status: PianoRollNotesStatus): boolean {
  return (
    status.status !== "absent" && ACTIVE_NOTES_STATUSES.has(status.status)
  );
}

export function usePianoRollNotesStatus(jobId: string | null) {
  return useQuery({
    queryKey: ["piano-roll-notes-status", jobId],
    queryFn: () => fetchPianoRollNotesStatus(jobId as string),
    enabled: jobId != null,
    refetchInterval: (query) => {
      const data = query.state.data as PianoRollNotesStatus | undefined;
      if (!data) return 3000;
      return isActiveNotesStatus(data) ? 3000 : false;
    },
  });
}

export function usePianoRollNotes(
  jobId: string | null,
  viewport: PianoRollNotesViewport,
  enabled: boolean,
) {
  return useQuery({
    queryKey: [
      "piano-roll-notes",
      jobId,
      viewport.startUtc ?? null,
      viewport.endUtc ?? null,
      viewport.extractorVersion ?? null,
      viewport.eventIds ? [...viewport.eventIds].sort().join(",") : null,
    ],
    queryFn: () => fetchPianoRollNotes(jobId as string, viewport),
    enabled: enabled && jobId != null,
  });
}

/**
 * 32-bit FNV-1a content hash over the joined uid sequence, returned as
 * an 8-char base-36 string. Used to key the React Query batch entry
 * without inlining ~74 KB of UUIDs into the queryKey. Collision
 * probability for unrelated viewports is ~1/2^32 — orders of magnitude
 * tighter than length-plus-first-and-last, which collides on any two
 * batches sharing those three fields but differing in their interior.
 */
function fnv1aHex(input: string): string {
  let h = 0x811c9dc5;
  for (let i = 0; i < input.length; i += 1) {
    h ^= input.charCodeAt(i);
    h = Math.imul(h, 0x01000193);
  }
  return (h >>> 0).toString(36);
}

/**
 * Batch-fetches v3 ribbon contours for the requested ``note_uid``s and
 * populates a per-uid React Query cache entry for each row returned.
 *
 * The batch query itself uses a 32-bit FNV-1a content hash of the
 * sorted-uid join as the queryKey, so the key doesn't inline ~74 KB of
 * UUIDs but still discriminates batches that differ only in their
 * interior. Per-uid cache entries — keyed
 * ``["piano-roll-note-contour", jobId, ev, uid]`` — let downstream
 * consumers read a single note's contour without re-fetching, and let
 * callers that already filter against this cache skip a network round
 * trip entirely. uid stability is guaranteed by the backend's
 * deterministic UUID v5 derivation (ADR-069 §6.2).
 */
export function usePianoRollNoteContours(
  jobId: string | null,
  noteUids: string[],
  enabled: boolean,
  extractorVersion?: string | null,
) {
  const qc = useQueryClient();
  const sortedUids = [...noteUids].sort();
  const batchKey = sortedUids.length
    ? `${sortedUids.length}:${fnv1aHex(sortedUids.join(","))}`
    : "empty";
  return useQuery({
    queryKey: [
      "piano-roll-note-contours",
      jobId,
      extractorVersion ?? null,
      batchKey,
    ],
    queryFn: async () => {
      const response = await fetchPianoRollNoteContours(jobId as string, {
        noteUids: sortedUids,
        extractorVersion,
      });
      for (const [uid, rows] of Object.entries(response.contours)) {
        qc.setQueryData(
          [
            "piano-roll-note-contour",
            jobId,
            extractorVersion ?? null,
            uid,
          ],
          rows,
        );
      }
      return response;
    },
    enabled: enabled && jobId != null && sortedUids.length > 0,
  });
}

export function useCreatePianoRollNotesJob(jobId: string | null) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (body: CreatePianoRollNotesJobRequest = {}) =>
      createPianoRollNotesJob(jobId as string, body),
    onSuccess: () => {
      if (jobId == null) return;
      qc.invalidateQueries({ queryKey: ["piano-roll-notes-status", jobId] });
      qc.invalidateQueries({ queryKey: ["event-encoder-timeline", jobId] });
    },
  });
}

const ACTIVE_MIDI_EXPORT_STATUSES = new Set<PianoRollMidiExportJobStatus>([
  "queued",
  "running",
]);

function isActiveMidiExportStatus(status: PianoRollMidiExportStatus): boolean {
  return (
    status.status !== "absent" &&
    ACTIVE_MIDI_EXPORT_STATUSES.has(status.status)
  );
}

export function usePianoRollMidiExportStatus(jobId: string | null) {
  return useQuery({
    queryKey: ["piano-roll-midi-export-status", jobId],
    queryFn: () => fetchPianoRollMidiExportStatus(jobId as string),
    enabled: jobId != null,
    refetchInterval: (query) => {
      const data = query.state.data as PianoRollMidiExportStatus | undefined;
      if (!data) return 3000;
      return isActiveMidiExportStatus(data) ? 3000 : false;
    },
  });
}

export function useCreatePianoRollMidiExport(jobId: string | null) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (body: CreatePianoRollMidiExportRequest) =>
      createPianoRollMidiExport(jobId as string, body),
    onSuccess: () => {
      if (jobId == null) return;
      qc.invalidateQueries({ queryKey: ["piano-roll-midi-export-status", jobId] });
    },
  });
}

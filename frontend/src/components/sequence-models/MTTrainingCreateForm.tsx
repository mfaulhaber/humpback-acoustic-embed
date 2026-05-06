import { useEffect, useMemo, useState } from "react";
import { Plus, Trash2 } from "lucide-react";
import { useNavigate } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  type ContinuousEmbeddingJob,
  type MaskedTransformerJobCreate,
  type MaskedTransformerPreset,
  type MaskedTransformerRetrievalHeadArch,
  type MaskedTransformerSequenceConstructionMode,
  continuousEmbeddingSourceKind,
  useContinuousEmbeddingJobs,
  useCreateMaskedTransformerJob,
  useEventClassificationJobsForSegmentation,
} from "@/api/sequenceModels";

const PRESETS: MaskedTransformerPreset[] = ["small", "default", "large"];
const RETRIEVAL_HEAD_ARCHES: MaskedTransformerRetrievalHeadArch[] = [
  "mlp",
  "linear",
];

interface SourceRowState {
  id: string;
  continuousEmbeddingJobId: string;
  eventClassificationJobId: string;
  sourceAlias: string;
}

function parseKValues(input: string): number[] | null {
  const values = input
    .split(/[\s,]+/)
    .map((token) => token.trim())
    .filter(Boolean);
  if (values.length === 0) return null;
  const out: number[] = [];
  for (const token of values) {
    const value = Number.parseInt(token, 10);
    if (!Number.isFinite(value) || value < 2) return null;
    if (!out.includes(value)) out.push(value);
  }
  return out;
}

function compatibleValue(value: unknown) {
  return typeof value === "number" ? Number(value.toFixed(9)) : value ?? null;
}

function sourceCompatibilityError(sources: ContinuousEmbeddingJob[]): string | null {
  if (sources.length <= 1) return null;
  const fields: (keyof ContinuousEmbeddingJob)[] = [
    "vector_dim",
    "model_version",
    "chunk_size_seconds",
    "chunk_hop_seconds",
    "projection_kind",
    "projection_dim",
  ];
  const first = sources[0];
  for (const field of fields) {
    const expected = compatibleValue(first[field]);
    const mismatch = sources.some((source) => compatibleValue(source[field]) !== expected);
    if (mismatch) return `Selected sources must share ${String(field)}.`;
  }
  const checkpoints = sources.map((source) => source.crnn_checkpoint_sha256);
  if (checkpoints.every(Boolean) && checkpoints.some((value) => value !== checkpoints[0])) {
    return "Selected sources must share crnn_checkpoint_sha256 when known.";
  }
  return null;
}

function SourcePairRow({
  row,
  index,
  eligible,
  canRemove,
  onChange,
  onRemove,
}: {
  row: SourceRowState;
  index: number;
  eligible: ContinuousEmbeddingJob[];
  canRemove: boolean;
  onChange: (patch: Partial<SourceRowState>) => void;
  onRemove: () => void;
}) {
  const selectedJob =
    eligible.find((job) => job.id === row.continuousEmbeddingJobId) ?? null;
  const segmentationJobId = selectedJob?.event_segmentation_job_id ?? null;
  const classifyJobsQuery = useEventClassificationJobsForSegmentation(
    segmentationJobId,
  );
  const classifyJobs = classifyJobsQuery.data ?? [];

  useEffect(() => {
    if (row.continuousEmbeddingJobId === "") return;
    if (classifyJobs.length === 0) {
      if (row.eventClassificationJobId !== "") {
        onChange({ eventClassificationJobId: "" });
      }
      return;
    }
    if (!classifyJobs.some((job) => job.id === row.eventClassificationJobId)) {
      onChange({ eventClassificationJobId: classifyJobs[0].id });
    }
  }, [
    classifyJobs,
    onChange,
    row.continuousEmbeddingJobId,
    row.eventClassificationJobId,
  ]);

  return (
    <div
      className="grid gap-2 rounded-md border p-3 md:grid-cols-[minmax(0,1.5fr)_minmax(0,1.3fr)_minmax(0,1fr)_auto]"
      data-testid={`mt-training-source-row-${index}`}
    >
      <label className="grid gap-1 text-xs font-medium">
        Embedding Job
        <select
          className="rounded-md border px-2 py-1 text-sm"
          data-testid={`mt-training-source-${index}`}
          value={row.continuousEmbeddingJobId}
          onChange={(event) =>
            onChange({
              continuousEmbeddingJobId: event.target.value,
              eventClassificationJobId: "",
            })
          }
        >
          <option value="">Select source</option>
          {eligible.map((job) => (
            <option key={job.id} value={job.id}>
              {job.id.slice(0, 8)} · vec={job.vector_dim ?? "?"} · chunks=
              {job.total_chunks ?? "?"}
            </option>
          ))}
        </select>
      </label>
      <label className="grid gap-1 text-xs font-medium">
        Classify Job
        <select
          className="rounded-md border px-2 py-1 text-sm"
          data-testid={`mt-training-classify-${index}`}
          value={row.eventClassificationJobId}
          disabled={
            row.continuousEmbeddingJobId === "" ||
            classifyJobsQuery.isLoading ||
            classifyJobs.length === 0
          }
          onChange={(event) =>
            onChange({ eventClassificationJobId: event.target.value })
          }
        >
          {classifyJobs.length === 0 ? <option value="">None</option> : null}
          {classifyJobs.map((job) => (
            <option key={job.id} value={job.id}>
              {job.id.slice(0, 8)}
              {job.model_name ? ` · ${job.model_name}` : ""}
              {job.n_events_classified != null
                ? ` · ${job.n_events_classified} events`
                : ""}
            </option>
          ))}
        </select>
      </label>
      <label className="grid gap-1 text-xs font-medium">
        Alias
        <Input
          data-testid={`mt-training-source-alias-${index}`}
          value={row.sourceAlias}
          onChange={(event) => onChange({ sourceAlias: event.target.value })}
        />
      </label>
      <div className="flex items-end">
        <Button
          type="button"
          variant="ghost"
          size="icon"
          disabled={!canRemove}
          onClick={onRemove}
          data-testid={`mt-training-remove-source-${index}`}
          title="Remove source"
        >
          <Trash2 className="h-4 w-4" />
        </Button>
      </div>
    </div>
  );
}

export function MTTrainingCreateForm() {
  const navigate = useNavigate();
  const { data: cejJobs = [] } = useContinuousEmbeddingJobs();
  const createMutation = useCreateMaskedTransformerJob();
  const eligible = useMemo(
    () =>
      cejJobs.filter(
        (job) =>
          job.status === "complete" &&
          continuousEmbeddingSourceKind(job) === "region_crnn",
      ),
    [cejJobs],
  );

  const [rows, setRows] = useState<SourceRowState[]>([
    {
      id: crypto.randomUUID(),
      continuousEmbeddingJobId: "",
      eventClassificationJobId: "",
      sourceAlias: "",
    },
  ]);
  const [preset, setPreset] = useState<MaskedTransformerPreset>("default");
  const [kInput, setKInput] = useState("100");
  const [maxEpochs, setMaxEpochs] = useState(30);
  const [maskFraction, setMaskFraction] = useState(0.2);
  const [spanMin, setSpanMin] = useState(2);
  const [spanMax, setSpanMax] = useState(6);
  const [dropout, setDropout] = useState(0.1);
  const [cosineWeight, setCosineWeight] = useState(0);
  const [batchSize, setBatchSize] = useState(8);
  const [retrievalHeadEnabled, setRetrievalHeadEnabled] = useState(false);
  const [retrievalHeadArch, setRetrievalHeadArch] =
    useState<MaskedTransformerRetrievalHeadArch>("mlp");
  const [retrievalDim, setRetrievalDim] = useState(128);
  const [retrievalHiddenDim, setRetrievalHiddenDim] = useState(512);
  const [retrievalL2Normalize, setRetrievalL2Normalize] = useState(true);
  const [sequenceMode, setSequenceMode] =
    useState<MaskedTransformerSequenceConstructionMode>("region");
  const [eventCenteredFraction, setEventCenteredFraction] = useState(0.5);
  const [preEventContextSec, setPreEventContextSec] = useState(2);
  const [postEventContextSec, setPostEventContextSec] = useState(2);
  const [earlyStop, setEarlyStop] = useState(3);
  const [valSplit, setValSplit] = useState(0.1);
  const [seed, setSeed] = useState(42);
  const [error, setError] = useState<string | null>(null);

  const kValues = parseKValues(kInput);
  const selectedSources = rows
    .map((row) =>
      eligible.find((job) => job.id === row.continuousEmbeddingJobId),
    )
    .filter((job): job is ContinuousEmbeddingJob => job != null);
  const compatibilityError = sourceCompatibilityError(selectedSources);
  const duplicatePairs = new Set<string>();
  let hasDuplicatePair = false;
  for (const row of rows) {
    if (!row.continuousEmbeddingJobId || !row.eventClassificationJobId) continue;
    const key = `${row.continuousEmbeddingJobId}:${row.eventClassificationJobId}`;
    if (duplicatePairs.has(key)) hasDuplicatePair = true;
    duplicatePairs.add(key);
  }
  const sourceRowsComplete = rows.every(
    (row) => row.continuousEmbeddingJobId && row.eventClassificationJobId,
  );
  const canSubmit =
    sourceRowsComplete &&
    !hasDuplicatePair &&
    compatibilityError == null &&
    kValues != null &&
    spanMax >= spanMin &&
    batchSize > 0 &&
    retrievalDim > 0 &&
    (retrievalHeadArch === "linear" || retrievalHiddenDim > 0) &&
    !createMutation.isPending;

  const updateRow = (id: string, patch: Partial<SourceRowState>) => {
    setRows((current) =>
      current.map((row) => (row.id === id ? { ...row, ...patch } : row)),
    );
  };

  const addRow = () => {
    setRows((current) => [
      ...current,
      {
        id: crypto.randomUUID(),
        continuousEmbeddingJobId: "",
        eventClassificationJobId: "",
        sourceAlias: "",
      },
    ]);
  };

  const submit = () => {
    setError(null);
    if (!canSubmit || kValues == null) return;
    const body: MaskedTransformerJobCreate = {
      sources: rows.map((row) => ({
        continuous_embedding_job_id: row.continuousEmbeddingJobId,
        event_classification_job_id: row.eventClassificationJobId,
        source_alias: row.sourceAlias || null,
      })),
      preset,
      k_values: kValues,
      max_epochs: maxEpochs,
      mask_fraction: maskFraction,
      span_length_min: spanMin,
      span_length_max: spanMax,
      dropout,
      cosine_loss_weight: cosineWeight,
      batch_size: batchSize,
      retrieval_head_enabled: retrievalHeadEnabled,
      retrieval_dim: retrievalHeadEnabled ? retrievalDim : null,
      retrieval_hidden_dim:
        retrievalHeadEnabled && retrievalHeadArch === "mlp"
          ? retrievalHiddenDim
          : null,
      retrieval_l2_normalize: retrievalL2Normalize,
      retrieval_head_arch: retrievalHeadEnabled ? retrievalHeadArch : "mlp",
      sequence_construction_mode: sequenceMode,
      event_centered_fraction:
        sequenceMode === "event_centered"
          ? 1
          : sequenceMode === "mixed"
            ? eventCenteredFraction
            : 0,
      pre_event_context_sec:
        sequenceMode === "region" ? null : preEventContextSec,
      post_event_context_sec:
        sequenceMode === "region" ? null : postEventContextSec,
      contrastive_loss_weight: 0,
      contrastive_label_source: "none",
      training_freeze_mode: "none",
      source_masked_transformer_job_id: null,
      negative_label_family_policy_json: null,
      early_stop_patience: earlyStop,
      val_split: valSplit,
      seed,
    };
    createMutation.mutate(body, {
      onSuccess: (job) => navigate(`/app/sequence-models/mt-training/${job.id}`),
      onError: (err) => {
        setError(err instanceof Error ? err.message : "Failed to create job");
      },
    });
  };

  return (
    <div className="space-y-4 rounded-md border p-4" data-testid="mt-training-create-form">
      <div className="flex items-center justify-between gap-3">
        <h2 className="text-base font-semibold">New MT Training Job</h2>
        <Button
          type="button"
          variant="outline"
          size="sm"
          onClick={addRow}
          data-testid="mt-training-add-source"
        >
          <Plus className="mr-2 h-4 w-4" />
          Source
        </Button>
      </div>

      <div className="space-y-2">
        {rows.map((row, index) => (
          <SourcePairRow
            key={row.id}
            row={row}
            index={index}
            eligible={eligible}
            canRemove={rows.length > 1}
            onChange={(patch) => updateRow(row.id, patch)}
            onRemove={() =>
              setRows((current) => current.filter((item) => item.id !== row.id))
            }
          />
        ))}
      </div>

      {eligible.length === 0 ? (
        <p className="text-xs text-muted-foreground">
          No completed CRNN region embedding jobs are available.
        </p>
      ) : null}
      {hasDuplicatePair ? (
        <p className="text-xs text-red-600" data-testid="mt-training-duplicate-error">
          Duplicate source pairs are not allowed.
        </p>
      ) : null}
      {compatibilityError ? (
        <p className="text-xs text-red-600" data-testid="mt-training-compat-error">
          {compatibilityError}
        </p>
      ) : null}

      <div className="grid gap-3 md:grid-cols-3">
        <label className="grid gap-1 text-xs font-medium">
          Preset
          <select
            className="rounded-md border px-2 py-1 text-sm"
            value={preset}
            onChange={(event) => setPreset(event.target.value as MaskedTransformerPreset)}
          >
            {PRESETS.map((item) => (
              <option key={item} value={item}>
                {item}
              </option>
            ))}
          </select>
        </label>
        <label className="grid gap-1 text-xs font-medium">
          k Values
          <Input
            data-testid="mt-training-k-values"
            value={kInput}
            onChange={(event) => setKInput(event.target.value)}
            className={kValues == null ? "border-red-500" : ""}
          />
        </label>
        <NumberField label="Max Epochs" value={maxEpochs} min={1} onChange={setMaxEpochs} />
        <NumberField label="Mask Fraction" value={maskFraction} step={0.05} onChange={setMaskFraction} />
        <NumberField label="Span Min" value={spanMin} min={1} onChange={(value) => setSpanMin(Math.round(value))} />
        <NumberField label="Span Max" value={spanMax} min={1} onChange={(value) => setSpanMax(Math.round(value))} />
        <NumberField label="Dropout" value={dropout} step={0.05} onChange={setDropout} />
        <NumberField label="Cosine Weight" value={cosineWeight} step={0.05} onChange={setCosineWeight} />
        <NumberField label="Batch Size" value={batchSize} min={1} onChange={(value) => setBatchSize(Math.round(value))} />
      </div>

      <div className="grid gap-3 md:grid-cols-3">
        <label className="flex items-center gap-2 text-sm">
          <input
            type="checkbox"
            data-testid="mt-training-retrieval-head-enabled"
            checked={retrievalHeadEnabled}
            onChange={(event) => setRetrievalHeadEnabled(event.target.checked)}
          />
          Retrieval head
        </label>
        <label className="grid gap-1 text-xs font-medium">
          Retrieval Head
          <select
            className="rounded-md border px-2 py-1 text-sm"
            disabled={!retrievalHeadEnabled}
            value={retrievalHeadArch}
            onChange={(event) =>
              setRetrievalHeadArch(
                event.target.value as MaskedTransformerRetrievalHeadArch,
              )
            }
          >
            {RETRIEVAL_HEAD_ARCHES.map((item) => (
              <option key={item} value={item}>
                {item}
              </option>
            ))}
          </select>
        </label>
        <label className="flex items-center gap-2 text-sm">
          <input
            type="checkbox"
            checked={retrievalL2Normalize}
            disabled={!retrievalHeadEnabled}
            onChange={(event) => setRetrievalL2Normalize(event.target.checked)}
          />
          L2 normalize
        </label>
        <NumberField label="Retrieval Dim" value={retrievalDim} min={1} disabled={!retrievalHeadEnabled} onChange={(value) => setRetrievalDim(Math.round(value))} />
        <NumberField label="Hidden Dim" value={retrievalHiddenDim} min={1} disabled={!retrievalHeadEnabled || retrievalHeadArch === "linear"} onChange={(value) => setRetrievalHiddenDim(Math.round(value))} />
      </div>

      <div className="grid gap-3 md:grid-cols-4">
        <label className="grid gap-1 text-xs font-medium">
          Sequence Mode
          <select
            className="rounded-md border px-2 py-1 text-sm"
            value={sequenceMode}
            onChange={(event) =>
              setSequenceMode(
                event.target.value as MaskedTransformerSequenceConstructionMode,
              )
            }
          >
            <option value="region">region</option>
            <option value="event_centered">event_centered</option>
            <option value="mixed">mixed</option>
          </select>
        </label>
        <NumberField label="Event Fraction" value={eventCenteredFraction} step={0.05} disabled={sequenceMode !== "mixed"} onChange={setEventCenteredFraction} />
        <NumberField label="Pre Event Context" value={preEventContextSec} step={0.25} disabled={sequenceMode === "region"} onChange={setPreEventContextSec} />
        <NumberField label="Post Event Context" value={postEventContextSec} step={0.25} disabled={sequenceMode === "region"} onChange={setPostEventContextSec} />
        <NumberField label="Early Stop" value={earlyStop} min={1} onChange={(value) => setEarlyStop(Math.round(value))} />
        <NumberField label="Val Split" value={valSplit} step={0.05} onChange={setValSplit} />
        <NumberField label="Seed" value={seed} onChange={(value) => setSeed(Math.round(value))} />
      </div>

      {error ? <p className="text-xs text-red-600">{error}</p> : null}
      <div className="flex justify-end">
        <Button
          type="button"
          onClick={submit}
          disabled={!canSubmit}
          data-testid="mt-training-submit"
        >
          Create Training Job
        </Button>
      </div>
    </div>
  );
}

function NumberField({
  label,
  value,
  min,
  step = 1,
  disabled = false,
  onChange,
}: {
  label: string;
  value: number;
  min?: number;
  step?: number;
  disabled?: boolean;
  onChange: (value: number) => void;
}) {
  return (
    <label className="grid gap-1 text-xs font-medium">
      {label}
      <Input
        type="number"
        min={min}
        step={step}
        disabled={disabled}
        value={value}
        onChange={(event) => onChange(Number(event.target.value))}
      />
    </label>
  );
}

import { useEffect, useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
  type MaskedTransformerJobCreate,
  type MaskedTransformerPreset,
  type MaskedTransformerSequenceConstructionMode,
  continuousEmbeddingSourceKind,
  useContinuousEmbeddingJobs,
  useCreateMaskedTransformerJob,
  useEventClassificationJobsForSegmentation,
} from "@/api/sequenceModels";

const PRESETS: MaskedTransformerPreset[] = ["small", "default", "large"];

function parseKValues(input: string): number[] | null {
  const tokens = input
    .split(/[\s,]+/)
    .map((t) => t.trim())
    .filter(Boolean);
  if (!tokens.length) return null;
  const out: number[] = [];
  for (const t of tokens) {
    const n = Number.parseInt(t, 10);
    if (!Number.isFinite(n) || n < 2) return null;
    if (!out.includes(n)) out.push(n);
  }
  return out;
}

export function MaskedTransformerCreateForm() {
  const { data: cejJobs = [] } = useContinuousEmbeddingJobs();
  const createMutation = useCreateMaskedTransformerJob();
  const navigate = useNavigate();

  const eligible = useMemo(
    () =>
      cejJobs.filter(
        (j) =>
          j.status === "complete" &&
          continuousEmbeddingSourceKind(j) === "region_crnn",
      ),
    [cejJobs],
  );

  const [sourceId, setSourceId] = useState("");
  const [classifyId, setClassifyId] = useState<string>("");
  const [preset, setPreset] = useState<MaskedTransformerPreset>("default");
  const [kInput, setKInput] = useState("100");
  const [maxEpochs, setMaxEpochs] = useState(30);
  const [maskWeightBias, setMaskWeightBias] = useState(true);

  // Advanced
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [maskFraction, setMaskFraction] = useState(0.2);
  const [spanMin, setSpanMin] = useState(2);
  const [spanMax, setSpanMax] = useState(6);
  const [dropout, setDropout] = useState(0.1);
  const [cosineWeight, setCosineWeight] = useState(0.0);
  const [batchSize, setBatchSize] = useState(8);
  const [retrievalHeadEnabled, setRetrievalHeadEnabled] = useState(false);
  const [retrievalDim, setRetrievalDim] = useState(128);
  const [retrievalHiddenDim, setRetrievalHiddenDim] = useState(512);
  const [retrievalL2Normalize, setRetrievalL2Normalize] = useState(true);
  const [sequenceMode, setSequenceMode] =
    useState<MaskedTransformerSequenceConstructionMode>("region");
  const [eventCenteredFraction, setEventCenteredFraction] = useState(0.5);
  const [preEventContextSec, setPreEventContextSec] = useState(2.0);
  const [postEventContextSec, setPostEventContextSec] = useState(2.0);
  const [contrastiveEnabled, setContrastiveEnabled] = useState(false);
  const [contrastiveWeight, setContrastiveWeight] = useState(0.1);
  const [contrastiveTemperature, setContrastiveTemperature] = useState(0.07);
  const [contrastiveMinEvents, setContrastiveMinEvents] = useState(4);
  const [contrastiveMinRegions, setContrastiveMinRegions] = useState(2);
  const [requireCrossRegionPositive, setRequireCrossRegionPositive] = useState(true);
  const [earlyStop, setEarlyStop] = useState(3);
  const [valSplit, setValSplit] = useState(0.1);
  const [seed, setSeed] = useState(42);

  const [error, setError] = useState<string | null>(null);

  const selectedJob = useMemo(
    () => eligible.find((j) => j.id === sourceId) ?? null,
    [eligible, sourceId],
  );
  const segmentationJobId = selectedJob?.event_segmentation_job_id ?? null;
  const classifyJobsQuery = useEventClassificationJobsForSegmentation(
    segmentationJobId,
  );
  const classifyJobs = classifyJobsQuery.data ?? [];

  useEffect(() => {
    if (classifyJobs.length === 0) {
      setClassifyId("");
      return;
    }
    if (!classifyJobs.some((c) => c.id === classifyId)) {
      setClassifyId(classifyJobs[0].id);
    }
  }, [classifyJobs, classifyId]);

  const kValues = parseKValues(kInput);
  const kValid = kValues !== null;
  const classifyDropdownReady =
    sourceId === "" || (!classifyJobsQuery.isLoading && classifyJobs.length > 0);
  const sequenceModeValid =
    sequenceMode !== "mixed" ||
    (eventCenteredFraction > 0.0 && eventCenteredFraction < 1.0);
  const contextValid =
    sequenceMode === "region" ||
    (preEventContextSec >= 0.0 && postEventContextSec >= 0.0);
  const contrastiveActive = retrievalHeadEnabled && contrastiveEnabled;
  const contrastiveValid =
    !contrastiveEnabled ||
    (retrievalHeadEnabled &&
      contrastiveWeight > 0.0 &&
      contrastiveTemperature > 0.0 &&
      contrastiveMinEvents > 0 &&
      contrastiveMinRegions > 0);
  const canSubmit =
    sourceId !== "" &&
    kValid &&
    !createMutation.isPending &&
    spanMax >= spanMin &&
    PRESETS.includes(preset) &&
    classifyDropdownReady &&
    classifyId !== "" &&
    sequenceModeValid &&
    contextValid &&
    contrastiveValid &&
    batchSize > 0 &&
    (!retrievalHeadEnabled || (retrievalDim > 0 && retrievalHiddenDim > 0));

  const handleSubmit = () => {
    setError(null);
    if (!canSubmit || !kValues) return;
    const body: MaskedTransformerJobCreate = {
      continuous_embedding_job_id: sourceId,
      event_classification_job_id: classifyId || undefined,
      preset,
      k_values: kValues,
      max_epochs: maxEpochs,
      mask_weight_bias: maskWeightBias,
      mask_fraction: maskFraction,
      span_length_min: spanMin,
      span_length_max: spanMax,
      dropout,
      cosine_loss_weight: cosineWeight,
      batch_size: batchSize,
      retrieval_head_enabled: retrievalHeadEnabled,
      retrieval_dim: retrievalHeadEnabled ? retrievalDim : null,
      retrieval_hidden_dim: retrievalHeadEnabled ? retrievalHiddenDim : null,
      retrieval_l2_normalize: retrievalL2Normalize,
      sequence_construction_mode: sequenceMode,
      event_centered_fraction:
        sequenceMode === "event_centered"
          ? 1.0
          : sequenceMode === "mixed"
            ? eventCenteredFraction
            : 0.0,
      pre_event_context_sec:
        sequenceMode === "region" ? null : preEventContextSec,
      post_event_context_sec:
        sequenceMode === "region" ? null : postEventContextSec,
      contrastive_loss_weight: contrastiveActive ? contrastiveWeight : 0.0,
      contrastive_temperature: contrastiveTemperature,
      contrastive_label_source: contrastiveActive ? "human_corrections" : "none",
      contrastive_min_events_per_label: contrastiveMinEvents,
      contrastive_min_regions_per_label: contrastiveMinRegions,
      require_cross_region_positive: requireCrossRegionPositive,
      related_label_policy_json: null,
      early_stop_patience: earlyStop,
      val_split: valSplit,
      seed,
    };
    createMutation.mutate(body, {
      onSuccess: (job) => {
        navigate(`/app/sequence-models/masked-transformer/${job.id}`);
      },
      onError: (err: unknown) => {
        const message =
          err instanceof Error ? err.message : "Failed to create job";
        setError(message);
      },
    });
  };

  return (
    <Card data-testid="masked-transformer-create-form">
      <CardHeader>
        <CardTitle className="text-base">New Masked Transformer Job</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="grid gap-2">
          <label className="text-xs font-medium" htmlFor="mt-source">
            Upstream embedding job (CRNN region-based)
          </label>
          <select
            id="mt-source"
            data-testid="mt-source-select"
            value={sourceId}
            onChange={(e) => setSourceId(e.target.value)}
            className="rounded-md border px-2 py-1 text-sm"
          >
            <option value="">— select —</option>
            {eligible.map((j) => (
              <option key={j.id} value={j.id}>
                {j.id.slice(0, 8)} · {j.model_version} · vec={j.vector_dim ?? "?"} · chunks={j.total_chunks ?? "?"}
              </option>
            ))}
          </select>
          {eligible.length === 0 && (
            <div className="text-xs text-muted-foreground">
              No completed CRNN region-based continuous-embedding jobs found.
            </div>
          )}
        </div>

        <div className="grid gap-2">
          <label className="text-xs font-medium" htmlFor="mt-classify">
            Event Classification Job
          </label>
          <select
            id="mt-classify"
            data-testid="mt-classify-select"
            value={classifyId}
            disabled={
              sourceId === "" ||
              classifyJobsQuery.isLoading ||
              classifyJobs.length === 0
            }
            onChange={(e) => setClassifyId(e.target.value)}
            className="rounded-md border px-2 py-1 text-sm"
          >
            {classifyJobs.length === 0 ? (
              <option value="">— none —</option>
            ) : null}
            {classifyJobs.map((c) => (
              <option key={c.id} value={c.id}>
                #{c.id.slice(0, 8)}
                {c.model_name ? ` · ${c.model_name}` : ""}
                {c.n_events_classified != null
                  ? ` · ${c.n_events_classified} events`
                  : ""}
              </option>
            ))}
          </select>
          {sourceId !== "" &&
            !classifyJobsQuery.isLoading &&
            classifyJobs.length === 0 ? (
            <p
              className="text-xs text-amber-700"
              data-testid="mt-classify-empty-helper"
            >
              Run Pass 3 Classify on this segmentation first
            </p>
          ) : null}
        </div>

        <div className="flex items-center gap-3" role="radiogroup" aria-label="Preset">
          <span className="text-xs font-medium">Preset</span>
          {PRESETS.map((p) => (
            <label
              key={p}
              className="text-sm flex items-center gap-1"
              data-testid={`mt-preset-${p}`}
            >
              <input
                type="radio"
                name="mt-preset"
                value={p}
                checked={preset === p}
                onChange={() => setPreset(p)}
              />
              {p}
            </label>
          ))}
        </div>

        <div className="grid grid-cols-2 gap-4">
          <div>
            <label className="text-xs font-medium" htmlFor="mt-k-values">
              k_values (comma-separated)
            </label>
            <Input
              id="mt-k-values"
              data-testid="mt-k-values"
              value={kInput}
              onChange={(e) => setKInput(e.target.value)}
              className={kValid ? "" : "border-red-500"}
            />
            {!kValid && (
              <div className="text-xs text-red-600">
                Each k must be an integer ≥ 2.
              </div>
            )}
          </div>
          <div>
            <label className="text-xs font-medium" htmlFor="mt-max-epochs">
              max_epochs
            </label>
            <Input
              id="mt-max-epochs"
              type="number"
              min={1}
              data-testid="mt-max-epochs"
              value={maxEpochs}
              onChange={(e) => setMaxEpochs(Number.parseInt(e.target.value || "1", 10))}
            />
          </div>
        </div>

        <label className="flex items-center gap-2 text-sm">
          <input
            type="checkbox"
            data-testid="mt-mask-weight-bias"
            checked={maskWeightBias}
            onChange={(e) => setMaskWeightBias(e.target.checked)}
          />
          mask-weight bias (weights event-adjacent positions higher)
        </label>

        <div>
          <button
            type="button"
            className="text-xs underline"
            data-testid="mt-show-advanced"
            onClick={() => setShowAdvanced((v) => !v)}
          >
            {showAdvanced ? "Hide advanced" : "Show advanced"}
          </button>
        </div>

        {showAdvanced && (
          <div
            className="grid grid-cols-2 gap-3 border rounded-md p-3"
            data-testid="mt-advanced-panel"
          >
            <Field label="mask_fraction" value={maskFraction} step={0.05} onChange={setMaskFraction} />
            <Field label="span_length_min" value={spanMin} step={1} onChange={(v) => setSpanMin(Math.max(1, Math.round(v)))} />
            <Field label="span_length_max" value={spanMax} step={1} onChange={(v) => setSpanMax(Math.max(1, Math.round(v)))} />
            <Field label="dropout" value={dropout} step={0.05} onChange={setDropout} />
            <Field label="cosine_loss_weight" value={cosineWeight} step={0.05} onChange={setCosineWeight} />
            <Field
              label="batch_size"
              value={batchSize}
              step={1}
              invalid={batchSize <= 0}
              onChange={(v) => setBatchSize(Math.max(1, Math.round(v)))}
            />
            <label className="col-span-2 flex items-center gap-2 text-sm">
              <input
                type="checkbox"
                data-testid="mt-retrieval-head-enabled"
                checked={retrievalHeadEnabled}
                onChange={(e) => {
                  setRetrievalHeadEnabled(e.target.checked);
                  if (!e.target.checked) setContrastiveEnabled(false);
                }}
              />
              retrieval projection head
            </label>
            <div
              className="col-span-2 grid gap-2"
              role="radiogroup"
              aria-label="Sequence construction"
            >
              <span className="text-xs font-medium">sequence construction</span>
              <div className="flex flex-wrap gap-3">
                {(["region", "event_centered", "mixed"] as const).map((mode) => (
                  <label
                    key={mode}
                    className="flex items-center gap-1 text-sm"
                    data-testid={`mt-sequence-mode-${mode}`}
                  >
                    <input
                      type="radio"
                      name="mt-sequence-mode"
                      value={mode}
                      checked={sequenceMode === mode}
                      onChange={() => setSequenceMode(mode)}
                    />
                    {mode}
                  </label>
                ))}
              </div>
            </div>
            <Field
              label="pre_event_context_sec"
              value={preEventContextSec}
              step={0.25}
              disabled={sequenceMode === "region"}
              invalid={sequenceMode !== "region" && preEventContextSec < 0}
              onChange={setPreEventContextSec}
            />
            <Field
              label="post_event_context_sec"
              value={postEventContextSec}
              step={0.25}
              disabled={sequenceMode === "region"}
              invalid={sequenceMode !== "region" && postEventContextSec < 0}
              onChange={setPostEventContextSec}
            />
            <Field
              label="event_centered_fraction"
              value={eventCenteredFraction}
              step={0.05}
              disabled={sequenceMode !== "mixed"}
              invalid={
                sequenceMode === "mixed" &&
                !(eventCenteredFraction > 0.0 && eventCenteredFraction < 1.0)
              }
              onChange={setEventCenteredFraction}
            />
            <Field
              label="retrieval_dim"
              value={retrievalDim}
              step={1}
              disabled={!retrievalHeadEnabled}
              invalid={retrievalHeadEnabled && retrievalDim <= 0}
              onChange={(v) => setRetrievalDim(Math.round(v))}
            />
            <Field
              label="retrieval_hidden_dim"
              value={retrievalHiddenDim}
              step={1}
              disabled={!retrievalHeadEnabled}
              invalid={retrievalHeadEnabled && retrievalHiddenDim <= 0}
              onChange={(v) => setRetrievalHiddenDim(Math.round(v))}
            />
            <label className="col-span-2 flex items-center gap-2 text-sm">
              <input
                type="checkbox"
                data-testid="mt-retrieval-l2-normalize"
                checked={retrievalL2Normalize}
                disabled={!retrievalHeadEnabled}
                onChange={(e) => setRetrievalL2Normalize(e.target.checked)}
              />
              L2 normalize retrieval embeddings
            </label>
            <label className="col-span-2 flex items-center gap-2 text-sm">
              <input
                type="checkbox"
                data-testid="mt-contrastive-enabled"
                checked={contrastiveEnabled}
                disabled={!retrievalHeadEnabled}
                onChange={(e) => setContrastiveEnabled(e.target.checked)}
              />
              human-correction contrastive loss
            </label>
            <Field
              label="contrastive_loss_weight"
              value={contrastiveWeight}
              step={0.05}
              disabled={!contrastiveActive}
              invalid={contrastiveActive && contrastiveWeight <= 0}
              onChange={setContrastiveWeight}
            />
            <Field
              label="contrastive_temperature"
              value={contrastiveTemperature}
              step={0.01}
              disabled={!contrastiveActive}
              invalid={contrastiveActive && contrastiveTemperature <= 0}
              onChange={setContrastiveTemperature}
            />
            <Field
              label="contrastive_min_events_per_label"
              value={contrastiveMinEvents}
              step={1}
              disabled={!contrastiveActive}
              invalid={contrastiveActive && contrastiveMinEvents <= 0}
              onChange={(v) => setContrastiveMinEvents(Math.max(1, Math.round(v)))}
            />
            <Field
              label="contrastive_min_regions_per_label"
              value={contrastiveMinRegions}
              step={1}
              disabled={!contrastiveActive}
              invalid={contrastiveActive && contrastiveMinRegions <= 0}
              onChange={(v) => setContrastiveMinRegions(Math.max(1, Math.round(v)))}
            />
            <label className="col-span-2 flex items-center gap-2 text-sm">
              <input
                type="checkbox"
                data-testid="mt-require-cross-region-positive"
                checked={requireCrossRegionPositive}
                disabled={!contrastiveActive}
                onChange={(e) => setRequireCrossRegionPositive(e.target.checked)}
              />
              prefer cross-region positives
            </label>
            <Field label="early_stop_patience" value={earlyStop} step={1} onChange={(v) => setEarlyStop(Math.max(1, Math.round(v)))} />
            <Field label="val_split" value={valSplit} step={0.05} onChange={setValSplit} />
            <Field label="seed" value={seed} step={1} onChange={(v) => setSeed(Math.round(v))} />
          </div>
        )}

        {error && (
          <div className="text-xs text-red-600" data-testid="mt-create-error">
            {error}
          </div>
        )}

        <div className="flex items-center justify-end">
          <Button
            data-testid="mt-create-submit"
            disabled={!canSubmit}
            onClick={handleSubmit}
          >
            {createMutation.isPending ? "Creating…" : "Create job"}
          </Button>
        </div>
      </CardContent>
    </Card>
  );
}

function Field({
  label,
  value,
  step,
  onChange,
  disabled = false,
  invalid = false,
}: {
  label: string;
  value: number;
  step: number;
  onChange: (v: number) => void;
  disabled?: boolean;
  invalid?: boolean;
}) {
  return (
    <div>
      <label className="text-xs font-medium" htmlFor={`mt-adv-${label}`}>
        {label}
      </label>
      <Input
        id={`mt-adv-${label}`}
        type="number"
        step={step}
        disabled={disabled}
        data-testid={`mt-adv-${label}`}
        className={invalid ? "border-red-500" : ""}
        value={value}
        onChange={(e) => {
          const next = Number.parseFloat(e.target.value);
          if (Number.isFinite(next)) onChange(next);
        }}
      />
    </div>
  );
}

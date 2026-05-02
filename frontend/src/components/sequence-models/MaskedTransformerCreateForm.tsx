import { useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
  type MaskedTransformerJobCreate,
  type MaskedTransformerPreset,
  continuousEmbeddingSourceKind,
  useContinuousEmbeddingJobs,
  useCreateMaskedTransformerJob,
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
  const [earlyStop, setEarlyStop] = useState(3);
  const [valSplit, setValSplit] = useState(0.1);
  const [seed, setSeed] = useState(42);

  const [error, setError] = useState<string | null>(null);

  const kValues = parseKValues(kInput);
  const kValid = kValues !== null;
  const canSubmit =
    sourceId !== "" &&
    kValid &&
    !createMutation.isPending &&
    spanMax >= spanMin &&
    PRESETS.includes(preset);

  const handleSubmit = () => {
    setError(null);
    if (!canSubmit || !kValues) return;
    const body: MaskedTransformerJobCreate = {
      continuous_embedding_job_id: sourceId,
      preset,
      k_values: kValues,
      max_epochs: maxEpochs,
      mask_weight_bias: maskWeightBias,
      mask_fraction: maskFraction,
      span_length_min: spanMin,
      span_length_max: spanMax,
      dropout,
      cosine_loss_weight: cosineWeight,
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
}: {
  label: string;
  value: number;
  step: number;
  onChange: (v: number) => void;
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
        value={value}
        onChange={(e) => {
          const next = Number.parseFloat(e.target.value);
          if (Number.isFinite(next)) onChange(next);
        }}
      />
    </div>
  );
}

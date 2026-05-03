import { useEffect, useMemo, useState } from "react";
import { Button } from "@/components/ui/button";
import {
  type MotifOccurrence,
  type MotifSummary,
  useCancelMotifExtractionJob,
  useCreateMotifExtractionJob,
  useMotifExtractionJobs,
  useMotifOccurrences,
  useMotifs,
} from "@/api/sequenceModels";
import { MotifExampleAlignment } from "./MotifExampleAlignment";

export interface MotifPanelSelection {
  motifKey: string | null;
  motif: MotifSummary | null;
  occurrences: MotifOccurrence[];
  occurrencesTotal: number;
  activeOccurrenceIndex: number;
}

function pct(v: number): string {
  return `${(v * 100).toFixed(0)}%`;
}

const DEFAULT_FORM = {
  min_ngram: 2,
  max_ngram: 8,
  minimum_occurrences: 5,
  minimum_event_sources: 2,
  frequency_weight: 0.4,
  event_source_weight: 0.3,
  event_core_weight: 0.2,
  low_background_weight: 0.1,
  call_probability_weight: "",
};

export type MotifExtractionPanelParent =
  | { kind: "hmm"; hmmSequenceJobId: string }
  | {
      kind: "masked_transformer";
      maskedTransformerJobId: string;
      k: number;
    };

export function MotifExtractionPanel({
  hmmSequenceJobId,
  regionDetectionJobId,
  onJumpToTimestamp,
  onPlayMotif,
  parent,
  onSelectionChange,
  activeOccurrenceIndex: controlledActiveIndex,
  onActiveOccurrenceChange,
  numLabels,
  hideRowHighlight,
  onUserSelectMotif,
}: {
  /** @deprecated — pass ``parent`` instead. Kept for backwards compatibility. */
  hmmSequenceJobId?: string;
  regionDetectionJobId: string;
  onJumpToTimestamp: (timestamp: number) => void;
  /** Optional motif-bounded playback handler forwarded to
   *  ``MotifExampleAlignment``. The masked-transformer page supplies this
   *  to route Play through the shared ``TimelinePlaybackHandle``; HMM
   *  callers omit it to keep the standalone-audio fallback. */
  onPlayMotif?: (occurrence: MotifOccurrence, idx: number) => void;
  parent?: MotifExtractionPanelParent;
  onSelectionChange?: (selection: MotifPanelSelection) => void;
  activeOccurrenceIndex?: number;
  onActiveOccurrenceChange?: (idx: number) => void;
  /** Total label count (HMM ``n_states`` or masked-transformer ``k``).
   *  Forwarded to ``MotifExampleAlignment`` so its swatches use the
   *  same palette the main timeline does. */
  numLabels?: number;
  /**
   * When ``true``, the row-highlight class is suppressed so the table
   * looks unselected even though an internal ``selectedMotif`` may
   * exist. Used by the masked-transformer page when its Token Count
   * (byLength) mode owns the timeline highlight.
   */
  hideRowHighlight?: boolean;
  /**
   * Fires every time the user clicks a motif row (regardless of whether
   * the selected motif key changed). The masked-transformer page uses
   * this to exit byLength mode when the reviewer picks a single motif
   * directly. Distinct from ``onSelectionChange``, which is driven by
   * derived state and may fire on initial load / refetch.
   */
  onUserSelectMotif?: (motifKey: string) => void;
}) {
  const resolvedParent: MotifExtractionPanelParent = parent ??
    (hmmSequenceJobId
      ? { kind: "hmm", hmmSequenceJobId }
      : ({ kind: "hmm", hmmSequenceJobId: "" } as const));

  const [form, setForm] = useState(DEFAULT_FORM);
  const [selectedMotif, setSelectedMotif] = useState<string | null>(null);
  const [internalActiveIndex, setInternalActiveIndex] = useState<number>(0);
  const isControlled = controlledActiveIndex != null;
  const activeOccurrenceIndex = isControlled
    ? controlledActiveIndex
    : internalActiveIndex;
  const setActiveOccurrenceIndex = (idx: number) => {
    if (!isControlled) setInternalActiveIndex(idx);
    onActiveOccurrenceChange?.(idx);
  };
  const { data: jobs = [] } = useMotifExtractionJobs(
    resolvedParent.kind === "hmm"
      ? { hmm_sequence_job_id: resolvedParent.hmmSequenceJobId }
      : {
          masked_transformer_job_id: resolvedParent.maskedTransformerJobId,
          parent_kind: "masked_transformer",
          k: resolvedParent.k,
        },
  );
  const createMutation = useCreateMotifExtractionJob();
  const cancelMutation = useCancelMotifExtractionJob();
  const activeJob = jobs[0] ?? null;
  const isComplete = activeJob?.status === "complete";
  const { data: motifsData } = useMotifs(activeJob?.id ?? null, 0, 100, isComplete);
  const motifs = motifsData?.items ?? [];
  const selected = useMemo<MotifSummary | null>(() => {
    if (selectedMotif == null) return motifs[0] ?? null;
    return motifs.find((m) => m.motif_key === selectedMotif) ?? motifs[0] ?? null;
  }, [motifs, selectedMotif]);
  const { data: occurrencesData } = useMotifOccurrences(
    activeJob?.id ?? null,
    selected?.motif_key ?? null,
    0,
    100,
    isComplete && selected != null,
  );

  useEffect(() => {
    if (selectedMotif == null && motifs.length > 0) {
      setSelectedMotif(motifs[0].motif_key);
    }
  }, [motifs, selectedMotif]);

  useEffect(() => {
    if (!isControlled) setInternalActiveIndex(0);
  }, [selected?.motif_key, isControlled]);

  const occurrences = occurrencesData?.items ?? [];
  const occurrencesTotal = occurrencesData?.total ?? occurrences.length;

  useEffect(() => {
    onSelectionChange?.({
      motifKey: selected?.motif_key ?? null,
      motif: selected,
      occurrences,
      occurrencesTotal,
      activeOccurrenceIndex,
    });
    // We intentionally include the data identity (occurrences) so callers
    // see fresh occurrence arrays after a refetch.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selected?.motif_key, occurrencesData, activeOccurrenceIndex]);

  const updateNumber = (key: keyof typeof DEFAULT_FORM, value: string) => {
    setForm((prev) => ({
      ...prev,
      [key]: key === "call_probability_weight" ? value : Number(value),
    }));
  };

  const create = () => {
    const baseConfig = {
      min_ngram: Number(form.min_ngram),
      max_ngram: Number(form.max_ngram),
      minimum_occurrences: Number(form.minimum_occurrences),
      minimum_event_sources: Number(form.minimum_event_sources),
      frequency_weight: Number(form.frequency_weight),
      event_source_weight: Number(form.event_source_weight),
      event_core_weight: Number(form.event_core_weight),
      low_background_weight: Number(form.low_background_weight),
      call_probability_weight:
        form.call_probability_weight === ""
          ? null
          : Number(form.call_probability_weight),
    };
    if (resolvedParent.kind === "hmm") {
      createMutation.mutate({
        ...baseConfig,
        parent_kind: "hmm",
        hmm_sequence_job_id: resolvedParent.hmmSequenceJobId,
      });
    } else {
      createMutation.mutate({
        ...baseConfig,
        parent_kind: "masked_transformer",
        masked_transformer_job_id: resolvedParent.maskedTransformerJobId,
        k: resolvedParent.k,
      });
    }
  };

  return (
    <div className="space-y-4" data-testid="motif-extraction-panel">
      <div className="flex items-center justify-between gap-3">
        <div className="text-sm text-slate-600">
          {activeJob
            ? `${activeJob.status} / ${activeJob.total_motifs ?? 0} motifs`
            : "No motif job"}
        </div>
        {activeJob &&
        (activeJob.status === "queued" || activeJob.status === "running") ? (
          <Button
            size="sm"
            variant="outline"
            disabled={cancelMutation.isPending}
            onClick={() => cancelMutation.mutate(activeJob.id)}
          >
            Cancel
          </Button>
        ) : null}
      </div>

      {(!activeJob || activeJob.status === "failed" || activeJob.status === "canceled") && (
        <div className="space-y-3" data-testid="motif-create-form">
          {activeJob?.error_message ? (
            <div className="text-sm text-red-700">{activeJob.error_message}</div>
          ) : null}
          <div className="grid grid-cols-4 gap-3 text-sm">
            <label className="space-y-1">
              <span className="text-slate-500">min n</span>
              <input
                className="w-full rounded border px-2 py-1"
                type="number"
                min={1}
                max={16}
                value={form.min_ngram}
                onChange={(e) => updateNumber("min_ngram", e.target.value)}
              />
            </label>
            <label className="space-y-1">
              <span className="text-slate-500">max n</span>
              <input
                className="w-full rounded border px-2 py-1"
                type="number"
                min={1}
                max={16}
                value={form.max_ngram}
                onChange={(e) => updateNumber("max_ngram", e.target.value)}
              />
            </label>
            <label className="space-y-1">
              <span className="text-slate-500">occurrences</span>
              <input
                className="w-full rounded border px-2 py-1"
                type="number"
                min={1}
                value={form.minimum_occurrences}
                onChange={(e) =>
                  updateNumber("minimum_occurrences", e.target.value)
                }
              />
            </label>
            <label className="space-y-1">
              <span className="text-slate-500">event sources</span>
              <input
                className="w-full rounded border px-2 py-1"
                type="number"
                min={1}
                value={form.minimum_event_sources}
                onChange={(e) =>
                  updateNumber("minimum_event_sources", e.target.value)
                }
              />
            </label>
          </div>
          <details className="rounded border p-3 text-sm">
            <summary className="cursor-pointer font-medium">Advanced</summary>
            <div className="mt-3 grid grid-cols-5 gap-3">
              {(
                [
                  ["frequency_weight", "frequency"],
                  ["event_source_weight", "event source"],
                  ["event_core_weight", "event core"],
                  ["low_background_weight", "low background"],
                  ["call_probability_weight", "call prob"],
                ] as const
              ).map(([key, label]) => (
                <label key={key} className="space-y-1">
                  <span className="text-slate-500">{label}</span>
                  <input
                    className="w-full rounded border px-2 py-1"
                    type="number"
                    min={0}
                    step={0.05}
                    value={form[key]}
                    onChange={(e) => updateNumber(key, e.target.value)}
                  />
                </label>
              ))}
            </div>
          </details>
          <Button
            size="sm"
            disabled={createMutation.isPending}
            onClick={create}
            data-testid="motif-create-submit"
          >
            Create
          </Button>
        </div>
      )}

      {activeJob && (activeJob.status === "queued" || activeJob.status === "running") ? (
        <div className="text-sm text-slate-500" data-testid="motif-running">
          Motif extraction is {activeJob.status}.
        </div>
      ) : null}

      {isComplete && (
        <div className="grid grid-cols-[minmax(0,1fr)_minmax(360px,0.8fr)] gap-4">
          <div className="overflow-x-auto">
            <table className="w-full text-xs" data-testid="motif-table">
              <thead>
                <tr className="text-left text-slate-500">
                  <th className="px-2 py-1">States</th>
                  <th className="px-2 py-1">Occ</th>
                  <th className="px-2 py-1">Events</th>
                  <th className="px-2 py-1">Core</th>
                  <th className="px-2 py-1">Bg</th>
                  <th className="px-2 py-1">Call</th>
                  <th className="px-2 py-1">Dur</th>
                  <th className="px-2 py-1">Rank</th>
                </tr>
              </thead>
              <tbody>
                {motifs.map((motif) => (
                  <tr
                    key={motif.motif_key}
                    className={`cursor-pointer border-t ${
                      !hideRowHighlight && selected?.motif_key === motif.motif_key
                        ? "bg-blue-50"
                        : ""
                    }`}
                    onClick={() => {
                      setSelectedMotif(motif.motif_key);
                      onUserSelectMotif?.(motif.motif_key);
                    }}
                  >
                    <td className="px-2 py-1 font-mono">{motif.motif_key}</td>
                    <td className="px-2 py-1">{motif.occurrence_count}</td>
                    <td className="px-2 py-1">{motif.event_source_count}</td>
                    <td className="px-2 py-1">{pct(motif.event_core_fraction)}</td>
                    <td className="px-2 py-1">{pct(motif.background_fraction)}</td>
                    <td className="px-2 py-1">
                      {motif.mean_call_probability == null
                        ? "-"
                        : motif.mean_call_probability.toFixed(2)}
                    </td>
                    <td className="px-2 py-1">
                      {motif.mean_duration_seconds.toFixed(2)}s
                    </td>
                    <td className="px-2 py-1">{motif.rank_score.toFixed(3)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <div
            className="max-h-[640px] overflow-y-auto"
            data-testid="motif-example-list"
          >
            <MotifExampleAlignment
              occurrences={occurrences}
              regionDetectionJobId={regionDetectionJobId}
              onJumpToTimestamp={onJumpToTimestamp}
              onPlayMotif={onPlayMotif}
              activeOccurrenceIndex={activeOccurrenceIndex}
              onActiveOccurrenceChange={setActiveOccurrenceIndex}
              numLabels={numLabels}
            />
          </div>
        </div>
      )}
    </div>
  );
}

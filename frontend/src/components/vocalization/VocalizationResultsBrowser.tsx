import { useState, useMemo, useRef } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Download, ChevronLeft, ChevronRight, Play, Pause } from "lucide-react";
import {
  useVocClassifierInferenceJob,
  useVocClassifierInferenceResults,
  useVocClassifierModel,
} from "@/hooks/queries/useVocalization";
import { vocClassifierInferenceExportUrl, detectionSpectrogramUrl, detectionAudioSliceUrl } from "@/api/client";
import type { VocClassifierPredictionRow } from "@/api/types";

const PAGE_SIZE = 50;

// Stable color palette for type badges
const TYPE_COLORS = [
  "bg-blue-100 text-blue-800 border-blue-200",
  "bg-green-100 text-green-800 border-green-200",
  "bg-purple-100 text-purple-800 border-purple-200",
  "bg-orange-100 text-orange-800 border-orange-200",
  "bg-pink-100 text-pink-800 border-pink-200",
  "bg-teal-100 text-teal-800 border-teal-200",
  "bg-yellow-100 text-yellow-800 border-yellow-200",
  "bg-indigo-100 text-indigo-800 border-indigo-200",
];

interface Props {
  jobId: string;
}

export function VocalizationResultsBrowser({ jobId }: Props) {
  const { data: job } = useVocClassifierInferenceJob(jobId);
  const { data: model } = useVocClassifierModel(job?.vocalization_model_id ?? null);

  const [page, setPage] = useState(0);
  const [thresholdOverrides, setThresholdOverrides] = useState<Record<string, number>>({});

  const vocabulary = model?.vocabulary_snapshot ?? [];
  const storedThresholds = model?.per_class_thresholds ?? {};

  // Effective thresholds: stored merged with overrides
  const effectiveThresholds = useMemo(() => {
    const t: Record<string, number> = { ...storedThresholds };
    for (const [k, v] of Object.entries(thresholdOverrides)) {
      t[k] = v;
    }
    return t;
  }, [storedThresholds, thresholdOverrides]);

  // Fetch results server-side (pagination), no server-side threshold filtering
  // We apply thresholds client-side for instant feedback
  const { data: rows = [], isLoading } = useVocClassifierInferenceResults(
    job?.status === "complete" ? jobId : null,
    { offset: page * PAGE_SIZE, limit: PAGE_SIZE },
  );

  // Build type→color map for stable coloring
  const typeColorMap = useMemo(() => {
    const m = new Map<string, string>();
    vocabulary.forEach((t, i) => {
      m.set(t, TYPE_COLORS[i % TYPE_COLORS.length]);
    });
    return m;
  }, [vocabulary]);

  if (!job || job.status !== "complete") {
    return (
      <Card>
        <CardContent className="py-8 text-center text-sm text-muted-foreground">
          {job?.status === "running" || job?.status === "queued"
            ? "Inference job is still processing..."
            : "Select a completed inference job to browse results."}
        </CardContent>
      </Card>
    );
  }

  const isDetectionSource = job.source_type === "detection_job";

  return (
    <Card>
      <CardHeader className="flex flex-row items-center justify-between pb-3">
        <CardTitle className="text-base">Results</CardTitle>
        <Button
          size="sm"
          variant="outline"
          onClick={() => {
            const url = vocClassifierInferenceExportUrl(jobId, effectiveThresholds);
            window.open(url, "_blank");
          }}
        >
          <Download className="h-3.5 w-3.5 mr-1" />
          Export TSV
        </Button>
      </CardHeader>
      <CardContent className="space-y-4">
        {/* Threshold sliders */}
        {vocabulary.length > 0 && (
          <div className="space-y-2">
            <h4 className="text-sm font-medium">Per-Type Thresholds</h4>
            <div className="grid gap-2 sm:grid-cols-2">
              {vocabulary.map((type) => {
                const value = effectiveThresholds[type] ?? 0.5;
                return (
                  <div key={type} className="flex items-center gap-2 text-sm">
                    <Badge variant="outline" className={`shrink-0 ${typeColorMap.get(type) ?? ""}`}>
                      {type}
                    </Badge>
                    <input
                      type="range"
                      min={0}
                      max={1}
                      step={0.01}
                      value={value}
                      onChange={(e) =>
                        setThresholdOverrides((prev) => ({
                          ...prev,
                          [type]: parseFloat(e.target.value),
                        }))
                      }
                      className="flex-1 h-1.5 accent-slate-600"
                    />
                    <span className="w-10 text-right text-xs text-muted-foreground tabular-nums">
                      {value.toFixed(2)}
                    </span>
                  </div>
                );
              })}
            </div>
          </div>
        )}

        {/* Result rows */}
        {isLoading ? (
          <p className="text-sm text-muted-foreground">Loading results...</p>
        ) : rows.length === 0 ? (
          <p className="text-sm text-muted-foreground">No results on this page.</p>
        ) : (
          <div className="border rounded-md divide-y">
            {rows.map((row, i) => (
              <PredictionRowItem
                key={`${row.filename}-${row.start_sec}-${i}`}
                row={row}
                thresholds={effectiveThresholds}
                typeColorMap={typeColorMap}
                detectionJobId={isDetectionSource ? job.source_id : null}
              />
            ))}
          </div>
        )}

        {/* Pagination */}
        <div className="flex items-center justify-between">
          <Button
            size="sm"
            variant="outline"
            disabled={page === 0}
            onClick={() => setPage((p) => p - 1)}
          >
            <ChevronLeft className="h-3.5 w-3.5 mr-1" />
            Prev
          </Button>
          <span className="text-sm text-muted-foreground">
            Page {page + 1}
            {rows.length < PAGE_SIZE && " (last)"}
          </span>
          <Button
            size="sm"
            variant="outline"
            disabled={rows.length < PAGE_SIZE}
            onClick={() => setPage((p) => p + 1)}
          >
            Next
            <ChevronRight className="h-3.5 w-3.5 ml-1" />
          </Button>
        </div>
      </CardContent>
    </Card>
  );
}

function PredictionRowItem({
  row,
  thresholds,
  typeColorMap,
  detectionJobId,
}: {
  row: VocClassifierPredictionRow;
  thresholds: Record<string, number>;
  typeColorMap: Map<string, string>;
  detectionJobId: string | null;
}) {
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const [playing, setPlaying] = useState(false);

  // Client-side threshold filtering
  const activeTags = Object.entries(row.scores)
    .filter(([type, score]) => score >= (thresholds[type] ?? 0.5))
    .sort(([, a], [, b]) => b - a);

  const hasUtc = row.start_utc != null && row.end_utc != null;
  const duration =
    row.start_sec != null && row.end_sec != null
      ? row.end_sec - row.start_sec
      : hasUtc
        ? row.end_utc! - row.start_utc!
        : 0;

  // Spectrogram + audio only available for detection job sources with UTC
  const canShowMedia = detectionJobId && hasUtc;
  const spectrogramSrc = canShowMedia
    ? detectionSpectrogramUrl(detectionJobId, row.start_utc!, duration)
    : null;
  const audioSrc = canShowMedia
    ? detectionAudioSliceUrl(detectionJobId, row.start_utc!, duration)
    : null;

  function togglePlay() {
    if (!audioRef.current) return;
    if (playing) {
      audioRef.current.pause();
    } else {
      audioRef.current.play();
    }
    setPlaying(!playing);
  }

  return (
    <div className="flex items-start gap-3 px-3 py-2">
      {/* Spectrogram thumbnail */}
      {spectrogramSrc ? (
        <img
          src={spectrogramSrc}
          alt="spectrogram"
          className="w-24 h-16 object-cover rounded border shrink-0"
          loading="lazy"
        />
      ) : (
        <div className="w-24 h-16 bg-muted rounded border flex items-center justify-center text-xs text-muted-foreground shrink-0">
          no preview
        </div>
      )}

      {/* Info + tags */}
      <div className="flex-1 min-w-0 space-y-1">
        <div className="flex items-center gap-2 text-sm">
          <span className="truncate font-mono text-xs">{row.filename}</span>
          <span className="text-xs text-muted-foreground shrink-0">
            {row.start_sec.toFixed(1)}s – {row.end_sec.toFixed(1)}s
          </span>
        </div>
        <div className="flex flex-wrap gap-1">
          {activeTags.length > 0 ? (
            activeTags.map(([type, score]) => (
              <Badge
                key={type}
                variant="outline"
                className={`text-xs ${typeColorMap.get(type) ?? ""}`}
              >
                {type} {(score * 100).toFixed(0)}%
              </Badge>
            ))
          ) : (
            <span className="text-xs text-muted-foreground italic">below threshold</span>
          )}
        </div>
      </div>

      {/* Audio playback button */}
      {audioSrc && (
        <div className="shrink-0">
          <Button
            size="icon"
            variant="ghost"
            className="h-8 w-8"
            onClick={togglePlay}
          >
            {playing ? (
              <Pause className="h-3.5 w-3.5" />
            ) : (
              <Play className="h-3.5 w-3.5" />
            )}
          </Button>
          <audio
            ref={audioRef}
            src={audioSrc}
            onEnded={() => setPlaying(false)}
            preload="none"
          />
        </div>
      )}
    </div>
  );
}

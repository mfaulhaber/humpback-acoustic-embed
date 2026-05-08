import { useMemo, useCallback, useRef } from "react";
import { useVocClusteringVisualization } from "@/hooks/queries/useVocalization";
import { detectionAudioSliceUrl } from "@/api/client";
import {
  ClusterProjectionPlot,
  type ClusterProjectionPlotPoint,
} from "@/components/shared/ClusterProjectionPlot";

const PALETTE = [
  "#3a86ff", "#e63946", "#2a9d8f", "#e9c46a", "#264653",
  "#f4a261", "#6c5ce7", "#00b894", "#fd79a8", "#636e72",
  "#d63031", "#74b9ff", "#a29bfe", "#ffeaa7", "#55efc4",
];

interface PointCustomData {
  detectionJobId: string;
  startUtc: number | null;
  category: string;
}

interface VocalizationUmapPlotProps {
  jobId: string;
}

export function VocalizationUmapPlot({ jobId }: VocalizationUmapPlotProps) {
  const { data: viz, isLoading } = useVocClusteringVisualization(jobId);
  const audioRef = useRef<HTMLAudioElement | null>(null);

  const points = useMemo(() => {
    if (!viz) return [] as ClusterProjectionPlotPoint<PointCustomData>[];

    const sortedLabels = [...new Set(viz.cluster_label)].sort((a, b) => a - b);
    const colors = new Map<number, string>();
    for (const [traceIdx, label] of sortedLabels.entries()) {
      colors.set(
        label,
        label === -1 ? "#b2bec3" : PALETTE[traceIdx % PALETTE.length],
      );
    }

    return viz.cluster_label.map((label, index) => {
      const isNoise = label === -1;
      return {
        id: `${viz.detection_job_id[index]}-${viz.embedding_row_index[index]}`,
        x: viz.x[index],
        y: viz.y[index],
        groupKey: String(label),
        groupLabel: isNoise ? "Noise" : `Cluster ${label}`,
        color: colors.get(label) ?? PALETTE[0],
        markerSize: isNoise ? 4 : 7,
        markerOpacity: isNoise ? 0.4 : 0.8,
        hoverText: viz.category[index] ?? "",
        customData: {
          detectionJobId: viz.detection_job_id[index],
          startUtc: viz.start_utc?.[index] ?? null,
          category: viz.category[index] ?? "",
        },
      };
    });
  }, [viz]);

  const handleClick = useCallback(
    (point: ClusterProjectionPlotPoint<PointCustomData>) => {
      const cd = point.customData;
      if (!cd?.detectionJobId || cd.startUtc == null) return;

      if (!audioRef.current) {
        audioRef.current = new Audio();
      }
      audioRef.current.src = detectionAudioSliceUrl(cd.detectionJobId, cd.startUtc, 5);
      audioRef.current.play().catch(() => {});
    },
    [],
  );

  if (isLoading) return <p className="text-sm text-muted-foreground">Loading UMAP data...</p>;
  if (!viz) return <p className="text-sm text-muted-foreground">No visualization data available.</p>;

  return (
    <ClusterProjectionPlot
      points={points}
      xAxisTitle="UMAP 1"
      yAxisTitle="UMAP 2"
      height={500}
      emptyMessage="No visualization data available."
      onPointClick={handleClick}
    />
  );
}

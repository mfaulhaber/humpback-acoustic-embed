import { useMemo, useState } from "react";

import {
  type EventEncoderJob,
  type EventEncoderProjectionMethod,
  type EventEncoderProjectionPoint,
  useEventEncoderProjection,
} from "@/api/sequenceModels";
import {
  ClusterProjectionPlot,
  type ClusterProjectionPlotPoint,
} from "@/components/shared/ClusterProjectionPlot";

import { labelColor } from "./constants";

interface EventEncoderClusterProjectionPanelProps {
  job: EventEncoderJob;
  selectedK: number;
  selectedEventId: string | null;
  onSelectEvent: (eventId: string) => void;
}

const METHODS: EventEncoderProjectionMethod[] = ["umap", "pca"];

export function EventEncoderClusterProjectionPanel({
  job,
  selectedK,
  selectedEventId,
  onSelectEvent,
}: EventEncoderClusterProjectionPanelProps) {
  const [method, setMethod] = useState<EventEncoderProjectionMethod>("umap");
  const isComplete = job.status === "complete";
  const { data, isLoading, error } = useEventEncoderProjection(
    job.id,
    selectedK,
    method,
    isComplete,
  );

  const points = useMemo(() => {
    if (!data) return [] as ClusterProjectionPlotPoint<EventEncoderProjectionPoint>[];
    return data.points.map((point) => ({
      id: point.event_id,
      x: point.x,
      y: point.y,
      groupKey: String(point.token_id),
      groupLabel: point.token_label,
      color: labelColor(point.token_id, Math.max(data.selected_k, 1)),
      hoverText: `${point.token_label} ${point.event_id}<br>confidence ${point.token_confidence.toFixed(3)}`,
      customData: point,
    }));
  }, [data]);

  return (
    <div data-testid="eej-cluster-projection-panel">
      <div className="mb-2 flex flex-wrap items-center gap-2">
        <h3 className="text-sm font-semibold">Cluster Projections</h3>
        <select
          className="ml-auto h-8 rounded-md border bg-background px-2 text-xs"
          value={method}
          onChange={(event) =>
            setMethod(event.target.value as EventEncoderProjectionMethod)
          }
          data-testid="eej-projection-method-select"
        >
          {METHODS.map((value) => (
            <option key={value} value={value}>
              {value.toUpperCase()}
            </option>
          ))}
        </select>
      </div>
      {!isComplete ? (
        <ProjectionMessage message="Cluster projections available after tokenization completes." />
      ) : isLoading ? (
        <ProjectionMessage message="Loading cluster projection..." />
      ) : error ? (
        <ProjectionMessage message="Cluster projection artifact is unavailable." />
      ) : !data || data.points.length === 0 ? (
        <ProjectionMessage message="No projection points are available." />
      ) : (
        <ClusterProjectionPlot
          points={points}
          xAxisTitle={data.x_axis_label}
          yAxisTitle={data.y_axis_label}
          height={340}
          selectedPointId={selectedEventId}
          testId="eej-cluster-projection-plot"
          onPointClick={(point) => onSelectEvent(point.id)}
        />
      )}
    </div>
  );
}

function ProjectionMessage({ message }: { message: string }) {
  return (
    <div
      className="rounded-md border border-dashed p-3 text-xs text-muted-foreground"
      data-testid="eej-cluster-projection-message"
    >
      {message}
    </div>
  );
}

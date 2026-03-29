import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
  Select,
  SelectContent,
  SelectGroup,
  SelectItem,
  SelectLabel,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { useMemo } from "react";
import { useQuery } from "@tanstack/react-query";
import {
  fetchDetectionJobs,
  fetchHydrophoneDetectionJobs,
  fetchHydrophones,
} from "@/api/client";
import type { DetectionJob, HydrophoneInfo } from "@/api/types";

interface Props {
  selectedJobId: string | null;
  onSelect: (jobId: string | null) => void;
}

function formatUtcDate(epoch: number): string {
  const d = new Date(epoch * 1000);
  return d.toISOString().replace("T", " ").slice(0, 16) + " UTC";
}

function formatJobLabel(
  job: DetectionJob,
  hydrophoneMap: Map<string, HydrophoneInfo>,
): string {
  if (job.hydrophone_id && job.start_timestamp && job.end_timestamp) {
    const info = hydrophoneMap.get(job.hydrophone_id);
    const displayName = info
      ? `${info.name} (${info.location})`
      : (job.hydrophone_name ?? job.hydrophone_id);
    return `${displayName}    ${formatUtcDate(job.start_timestamp)} — ${formatUtcDate(job.end_timestamp)}`;
  }
  const windowCount =
    (job.result_summary as Record<string, unknown> | null)?.n_total_windows ??
    (job.result_summary as Record<string, unknown> | null)?.n_detections ??
    null;
  const suffix = windowCount !== null ? `    ${windowCount} windows` : "";
  return `${job.audio_folder ?? "unknown"}${suffix}`;
}

export function DetectionJobPicker({ selectedJobId, onSelect }: Props) {
  const { data: localJobs = [] } = useQuery({
    queryKey: ["detectionJobs"],
    queryFn: fetchDetectionJobs,
  });
  const { data: hydrophoneJobs = [] } = useQuery({
    queryKey: ["hydrophoneDetectionJobs"],
    queryFn: fetchHydrophoneDetectionJobs,
  });
  const { data: hydrophones = [] } = useQuery({
    queryKey: ["hydrophones"],
    queryFn: fetchHydrophones,
    staleTime: 60_000,
  });

  const hydrophoneMap = useMemo(() => {
    const m = new Map<string, HydrophoneInfo>();
    for (const h of hydrophones) m.set(h.id, h);
    return m;
  }, [hydrophones]);

  const allJobs = [...localJobs, ...hydrophoneJobs];
  const completedJobs = allJobs.filter((j) => j.status === "complete");
  const hydrophone = completedJobs.filter((j) => j.hydrophone_id);
  const local = completedJobs.filter((j) => !j.hydrophone_id);

  return (
    <Card>
      <CardHeader className="pb-3">
        <CardTitle className="text-base">Source</CardTitle>
      </CardHeader>
      <CardContent>
        <Select
          value={selectedJobId ?? ""}
          onValueChange={(v) => onSelect(v || null)}
        >
          <SelectTrigger className="h-9">
            <SelectValue placeholder="Select a detection job..." />
          </SelectTrigger>
          <SelectContent>
            {hydrophone.length > 0 && (
              <SelectGroup>
                <SelectLabel>Hydrophone</SelectLabel>
                {hydrophone.map((j) => (
                  <SelectItem key={j.id} value={j.id}>
                    {formatJobLabel(j, hydrophoneMap)}
                  </SelectItem>
                ))}
              </SelectGroup>
            )}
            {local.length > 0 && (
              <SelectGroup>
                <SelectLabel>Local</SelectLabel>
                {local.map((j) => (
                  <SelectItem key={j.id} value={j.id}>
                    {formatJobLabel(j, hydrophoneMap)}
                  </SelectItem>
                ))}
              </SelectGroup>
            )}
          </SelectContent>
        </Select>
      </CardContent>
    </Card>
  );
}

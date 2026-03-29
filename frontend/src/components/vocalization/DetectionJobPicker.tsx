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
import { useQuery } from "@tanstack/react-query";
import { fetchDetectionJobs } from "@/api/client";
import type { DetectionJob } from "@/api/types";

interface Props {
  selectedJobId: string | null;
  onSelect: (jobId: string | null) => void;
}

function formatUtcDate(epoch: number): string {
  const d = new Date(epoch * 1000);
  return d.toISOString().replace("T", " ").slice(0, 16) + " UTC";
}

function formatJobLabel(job: DetectionJob): string {
  if (job.hydrophone_name && job.start_timestamp && job.end_timestamp) {
    return `${job.hydrophone_name}    ${formatUtcDate(job.start_timestamp)} — ${formatUtcDate(job.end_timestamp)}`;
  }
  const windowCount =
    (job.result_summary as Record<string, unknown> | null)?.n_total_windows ??
    (job.result_summary as Record<string, unknown> | null)?.n_detections ??
    null;
  const suffix = windowCount !== null ? `    ${windowCount} windows` : "";
  return `${job.audio_folder ?? "unknown"}${suffix}`;
}

export function DetectionJobPicker({ selectedJobId, onSelect }: Props) {
  const { data: jobs = [] } = useQuery({
    queryKey: ["detectionJobs"],
    queryFn: fetchDetectionJobs,
  });

  const completedJobs = jobs.filter((j) => j.status === "complete");
  const hydrophone = completedJobs.filter((j) => j.hydrophone_name);
  const local = completedJobs.filter((j) => !j.hydrophone_name);

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
                    {formatJobLabel(j)}
                  </SelectItem>
                ))}
              </SelectGroup>
            )}
            {local.length > 0 && (
              <SelectGroup>
                <SelectLabel>Local</SelectLabel>
                {local.map((j) => (
                  <SelectItem key={j.id} value={j.id}>
                    {formatJobLabel(j)}
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

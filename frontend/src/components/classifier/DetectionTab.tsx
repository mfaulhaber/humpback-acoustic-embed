import { useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import {
  useClassifierModels,
  useDetectionJobs,
  useCreateDetectionJob,
} from "@/hooks/queries/useClassifier";
import { detectionTsvUrl } from "@/api/client";
import type { DetectionJob } from "@/api/types";

export function DetectionTab() {
  const { data: models = [] } = useClassifierModels();
  const { data: detectionJobs = [] } = useDetectionJobs(3000);
  const createMutation = useCreateDetectionJob();

  const [selectedModelId, setSelectedModelId] = useState("");
  const [audioFolder, setAudioFolder] = useState("");
  const [threshold, setThreshold] = useState(0.5);

  const hasActiveJobs = detectionJobs.some(
    (j) => j.status === "queued" || j.status === "running"
  );

  const handleSubmit = () => {
    if (!selectedModelId || !audioFolder) return;
    createMutation.mutate(
      {
        classifier_model_id: selectedModelId,
        audio_folder: audioFolder,
        confidence_threshold: threshold,
      },
      {
        onSuccess: () => {
          setAudioFolder("");
        },
      }
    );
  };

  return (
    <div className="space-y-4">
      <Card>
        <CardHeader>
          <CardTitle className="text-base">Run Detection</CardTitle>
        </CardHeader>
        <CardContent className="space-y-3">
          <div>
            <label className="text-sm font-medium">Classifier Model</label>
            <select
              className="w-full border rounded px-3 py-2 text-sm mt-1"
              value={selectedModelId}
              onChange={(e) => setSelectedModelId(e.target.value)}
            >
              <option value="">Select a model…</option>
              {models.map((m) => (
                <option key={m.id} value={m.id}>
                  {m.name} ({m.model_version})
                </option>
              ))}
            </select>
            {models.length === 0 && (
              <p className="text-xs text-muted-foreground mt-1">
                No trained models. Train a classifier first.
              </p>
            )}
          </div>
          <div>
            <label className="text-sm font-medium">Audio Folder Path</label>
            <Input
              value={audioFolder}
              onChange={(e) => setAudioFolder(e.target.value)}
              placeholder="/path/to/hydrophone/recordings"
            />
          </div>
          <div>
            <label className="text-sm font-medium">
              Confidence Threshold: {threshold.toFixed(2)}
            </label>
            <input
              type="range"
              min="0"
              max="1"
              step="0.01"
              value={threshold}
              onChange={(e) => setThreshold(parseFloat(e.target.value))}
              className="w-full mt-1"
            />
          </div>
          <Button
            onClick={handleSubmit}
            disabled={
              !selectedModelId || !audioFolder || createMutation.isPending
            }
          >
            {createMutation.isPending ? "Creating…" : "Start Detection"}
          </Button>
          {createMutation.isError && (
            <p className="text-sm text-red-600">
              {(createMutation.error as Error).message}
            </p>
          )}
        </CardContent>
      </Card>

      {detectionJobs.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle className="text-base">
              Detection Jobs{" "}
              {hasActiveJobs && (
                <span className="text-xs text-muted-foreground">
                  (polling…)
                </span>
              )}
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-2">
              {detectionJobs.map((job) => (
                <DetectionJobRow key={job.id} job={job} />
              ))}
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
}

function DetectionJobRow({ job }: { job: DetectionJob }) {
  const statusColor: Record<string, string> = {
    queued: "bg-yellow-100 text-yellow-800",
    running: "bg-blue-100 text-blue-800",
    complete: "bg-green-100 text-green-800",
    failed: "bg-red-100 text-red-800",
    canceled: "bg-gray-100 text-gray-800",
  };

  const summary = job.result_summary as Record<string, number> | null;

  return (
    <div className="flex items-center justify-between p-2 border rounded text-sm">
      <div className="flex items-center gap-2">
        <Badge className={statusColor[job.status] ?? ""}>
          {job.status}
        </Badge>
        <span className="truncate max-w-64">{job.audio_folder}</span>
        <span className="text-muted-foreground">
          threshold={job.confidence_threshold}
        </span>
        {summary && (
          <span className="text-muted-foreground">
            {summary.n_spans} span(s) in {summary.n_files} file(s)
          </span>
        )}
      </div>
      <div className="flex items-center gap-2">
        {job.status === "complete" && job.output_tsv_path && (
          <a
            href={detectionTsvUrl(job.id)}
            download
            className="text-blue-600 hover:underline text-xs"
          >
            Download TSV
          </a>
        )}
        {job.error_message && (
          <span className="text-red-600 text-xs truncate max-w-48">
            {job.error_message}
          </span>
        )}
      </div>
    </div>
  );
}

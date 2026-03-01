import { useCallback } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { X } from "lucide-react";
import { FolderTree } from "@/components/shared/FolderTree";
import { StatusBadge } from "@/components/shared/StatusBadge";
import { useCancelProcessingJob } from "@/hooks/queries/useProcessing";
import { useAudioFiles } from "@/hooks/queries/useAudioFiles";
import { shortId, fmtDate } from "@/utils/format";
import { showMsg } from "@/components/shared/MessageToast";
import type { ProcessingJob } from "@/api/types";

interface ProcessingJobsListProps {
  jobs: ProcessingJob[];
}

export function ProcessingJobsList({ jobs }: ProcessingJobsListProps) {
  const cancelJob = useCancelProcessingJob();
  const { data: audioFiles = [] } = useAudioFiles();

  const audioMap = new Map(audioFiles.map((af) => [af.id, af]));

  const handleCancel = useCallback(
    (jobId: string) => {
      cancelJob.mutate(jobId, {
        onSuccess: () => showMsg("success", "Job canceled"),
        onError: (e) => showMsg("error", `Cancel failed: ${e.message}`),
      });
    },
    [cancelJob],
  );

  // Enrich jobs with audio info for folder tree
  type EnrichedJob = ProcessingJob & { _folderPath: string; _filename: string };
  const enriched: EnrichedJob[] = jobs.map((j) => {
    const af = audioMap.get(j.audio_file_id);
    return {
      ...j,
      _folderPath: af?.folder_path ?? "",
      _filename: af?.filename ?? j.audio_file_id,
    };
  });

  return (
    <Card>
      <CardHeader className="pb-3">
        <CardTitle className="text-base">Processing Jobs ({jobs.length})</CardTitle>
      </CardHeader>
      <CardContent>
        {jobs.length === 0 ? (
          <p className="text-sm text-muted-foreground">No processing jobs yet.</p>
        ) : (
          <FolderTree
            items={enriched}
            getPath={(j) => j._folderPath}
            stateKey="procTree"
            renderLeaf={(job) => (
              <div className="flex items-center gap-2 py-1 px-2 text-sm hover:bg-accent rounded">
                <span className="font-mono text-xs text-muted-foreground">{shortId(job.id)}</span>
                <span className="truncate">{job._filename}</span>
                <StatusBadge status={job.status} />
                <span className="text-xs text-muted-foreground">{job.model_version}</span>
                <span className="text-xs text-muted-foreground">{job.window_size_seconds}s</span>
                <span className="text-xs text-muted-foreground ml-auto">{fmtDate(job.created_at)}</span>
                {(job.status === "queued" || job.status === "running") && (
                  <Button
                    variant="ghost"
                    size="icon"
                    className="h-6 w-6 shrink-0"
                    onClick={() => handleCancel(job.id)}
                  >
                    <X className="h-3 w-3" />
                  </Button>
                )}
                {job.warning_message && (
                  <span className="text-yellow-600 text-xs">{job.warning_message}</span>
                )}
                {job.error_message && <span className="text-red-600 text-xs">{job.error_message}</span>}
              </div>
            )}
          />
        )}
      </CardContent>
    </Card>
  );
}

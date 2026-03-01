import { useState, useCallback } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { useAudioFiles } from "@/hooks/queries/useAudioFiles";
import { useModels } from "@/hooks/queries/useAdmin";
import { useCreateProcessingJob } from "@/hooks/queries/useProcessing";
import { createProcessingJob, fetchAudioFiles, fetchProcessingJobs } from "@/api/client";
import { showMsg } from "@/components/shared/MessageToast";
import { audioDisplayName } from "@/utils/format";
import type { ProcessingJob } from "@/api/types";

interface QueueJobFormProps {
  jobs: ProcessingJob[];
}

export function QueueJobForm({ jobs }: QueueJobFormProps) {
  const { data: audioFiles = [] } = useAudioFiles();
  const { data: models = [] } = useModels();
  const createJob = useCreateProcessingJob();

  const [audioFileId, setAudioFileId] = useState("");
  const [modelVersion, setModelVersion] = useState("");
  const [windowSize, setWindowSize] = useState("5");
  const [sampleRate, setSampleRate] = useState("32000");
  const [batchProgress, setBatchProgress] = useState<{
    done: number;
    total: number;
    skipped: number;
    failed: number;
  } | null>(null);

  const defaultModel = models.find((m) => m.is_default);

  const handleQueueJob = useCallback(() => {
    if (!audioFileId) {
      showMsg("error", "Please select an audio file");
      return;
    }
    createJob.mutate(
      {
        audio_file_id: audioFileId,
        model_version: modelVersion || defaultModel?.name || undefined,
        window_size_seconds: parseFloat(windowSize) || 5,
        target_sample_rate: parseInt(sampleRate) || 32000,
      },
      {
        onSuccess: (job) => {
          if (job.skipped) {
            showMsg("warning", "Job skipped — embedding already exists");
          } else {
            showMsg("success", "Processing job queued");
          }
        },
        onError: (e) => showMsg("error", `Failed to queue job: ${e.message}`),
      },
    );
  }, [audioFileId, modelVersion, windowSize, sampleRate, defaultModel, createJob]);

  const handleQueueAll = useCallback(async () => {
    const mv = modelVersion || defaultModel?.name || "";
    const ws = parseFloat(windowSize) || 5;
    const sr = parseInt(sampleRate) || 32000;

    const [allAudio, allJobs] = await Promise.all([fetchAudioFiles(), fetchProcessingJobs()]);

    // Filter to unprocessed
    const unprocessed = allAudio.filter((af) => {
      return !allJobs.some(
        (j) =>
          j.audio_file_id === af.id &&
          j.model_version === mv &&
          j.window_size_seconds === ws &&
          j.target_sample_rate === sr &&
          j.status !== "failed",
      );
    });

    if (unprocessed.length === 0) {
      showMsg("warning", "All audio files already have jobs for these settings");
      return;
    }

    setBatchProgress({ done: 0, total: unprocessed.length, skipped: 0, failed: 0 });
    let done = 0;
    let skipped = 0;
    let failed = 0;

    for (const af of unprocessed) {
      try {
        const job = await createProcessingJob({
          audio_file_id: af.id,
          model_version: mv || undefined,
          window_size_seconds: ws,
          target_sample_rate: sr,
        });
        if (job.skipped) skipped++;
      } catch {
        failed++;
      }
      done++;
      setBatchProgress({ done, total: unprocessed.length, skipped, failed });
    }

    setBatchProgress(null);
    showMsg(
      "success",
      `Queued ${done - skipped - failed} job(s), ${skipped} skipped, ${failed} failed`,
    );
  }, [modelVersion, windowSize, sampleRate, defaultModel]);

  const sortedAudio = [...audioFiles].sort((a, b) => {
    const da = audioDisplayName(a.filename, a.folder_path);
    const db = audioDisplayName(b.filename, b.folder_path);
    return da.localeCompare(db);
  });

  const batchPct =
    batchProgress && batchProgress.total > 0
      ? Math.round((batchProgress.done / batchProgress.total) * 100)
      : 0;

  return (
    <Card>
      <CardHeader className="pb-3">
        <CardTitle className="text-base">Queue Processing Job</CardTitle>
      </CardHeader>
      <CardContent className="space-y-3">
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3">
          <div>
            <label className="text-xs text-muted-foreground">Audio File</label>
            <Select value={audioFileId} onValueChange={setAudioFileId}>
              <SelectTrigger className="h-9">
                <SelectValue placeholder="Select audio file..." />
              </SelectTrigger>
              <SelectContent>
                {sortedAudio.map((af) => (
                  <SelectItem key={af.id} value={af.id}>
                    {audioDisplayName(af.filename, af.folder_path)}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
          <div>
            <label className="text-xs text-muted-foreground">Model Version</label>
            <Select value={modelVersion} onValueChange={setModelVersion}>
              <SelectTrigger className="h-9">
                <SelectValue placeholder={defaultModel ? `${defaultModel.name} (default)` : "Select..."} />
              </SelectTrigger>
              <SelectContent>
                {models.map((m) => (
                  <SelectItem key={m.id} value={m.name}>
                    {m.display_name || m.name}
                    {m.is_default ? " (default)" : ""}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
          <div>
            <label className="text-xs text-muted-foreground">Window Size (s)</label>
            <Input
              type="number"
              value={windowSize}
              onChange={(e) => setWindowSize(e.target.value)}
              className="h-9"
              min={0.1}
              step={0.1}
            />
          </div>
          <div>
            <label className="text-xs text-muted-foreground">Sample Rate (Hz)</label>
            <Input
              type="number"
              value={sampleRate}
              onChange={(e) => setSampleRate(e.target.value)}
              className="h-9"
              min={1000}
              step={1000}
            />
          </div>
        </div>

        <div className="flex gap-2">
          <Button size="sm" onClick={handleQueueJob} disabled={createJob.isPending || !!batchProgress}>
            Queue Job
          </Button>
          <Button
            size="sm"
            variant="secondary"
            onClick={handleQueueAll}
            disabled={createJob.isPending || !!batchProgress}
          >
            Queue All Unprocessed
          </Button>
        </div>

        {batchProgress && (
          <div>
            <div className="w-full bg-secondary rounded-full h-2">
              <div className="bg-primary h-2 rounded-full transition-all" style={{ width: `${batchPct}%` }} />
            </div>
            <p className="text-xs text-muted-foreground mt-1">
              Queued {batchProgress.done} / {batchProgress.total}
              {batchProgress.skipped > 0 && ` (${batchProgress.skipped} skipped)`}
              {batchProgress.failed > 0 && ` (${batchProgress.failed} failed)`}
            </p>
          </div>
        )}
      </CardContent>
    </Card>
  );
}

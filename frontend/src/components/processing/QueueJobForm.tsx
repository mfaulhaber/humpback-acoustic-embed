import { useState, useCallback, useMemo } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { useAudioFiles } from "@/hooks/queries/useAudioFiles";
import { useModels } from "@/hooks/queries/useAdmin";
import { useCreateProcessingJob } from "@/hooks/queries/useProcessing";
import { createProcessingJob, fetchAudioFiles, fetchProcessingJobs } from "@/api/client";
import { showMsg } from "@/components/shared/MessageToast";
import type { AudioFile, ProcessingJob } from "@/api/types";

const ROOT_SENTINEL = "__root__";

interface QueueJobFormProps {
  jobs: ProcessingJob[];
  onModelUsed?: (model: string) => void;
}

export function QueueJobForm({ jobs, onModelUsed }: QueueJobFormProps) {
  const { data: audioFiles = [] } = useAudioFiles();
  const { data: models = [] } = useModels();
  const createJob = useCreateProcessingJob();

  const [selectedFolder, setSelectedFolder] = useState("");
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

  // Group audio files by top-level parent folder
  const folderGroups = useMemo(() => {
    const groups = new Map<string, AudioFile[]>();
    for (const af of audioFiles) {
      const parent = af.folder_path ? af.folder_path.split("/")[0] : ROOT_SENTINEL;
      const list = groups.get(parent);
      if (list) {
        list.push(af);
      } else {
        groups.set(parent, [af]);
      }
    }
    return groups;
  }, [audioFiles]);

  // Sorted folder keys for the dropdown
  const sortedFolders = useMemo(() => {
    const keys = [...folderGroups.keys()];
    keys.sort((a, b) => {
      if (a === ROOT_SENTINEL) return -1;
      if (b === ROOT_SENTINEL) return 1;
      return a.localeCompare(b);
    });
    return keys;
  }, [folderGroups]);

  // "Already processed" status for selected folder
  const folderStatus = useMemo(() => {
    if (!selectedFolder) return null;
    const files = folderGroups.get(selectedFolder);
    if (!files || files.length === 0) return null;

    const mv = modelVersion || defaultModel?.name || "";
    const ws = parseFloat(windowSize) || 5;
    const sr = parseInt(sampleRate) || 32000;

    const unprocessedCount = files.filter((af) => {
      return !jobs.some(
        (j) =>
          j.audio_file_id === af.id &&
          j.model_version === mv &&
          j.window_size_seconds === ws &&
          j.target_sample_rate === sr &&
          j.status !== "failed",
      );
    }).length;

    return { total: files.length, unprocessed: unprocessedCount };
  }, [selectedFolder, folderGroups, jobs, modelVersion, windowSize, sampleRate, defaultModel]);

  const folderDisplayName = (key: string) => (key === ROOT_SENTINEL ? "(root)" : key);

  const handleQueueFolder = useCallback(async () => {
    if (!selectedFolder) {
      showMsg("error", "Please select an audio folder");
      return;
    }

    const mv = modelVersion || defaultModel?.name || "";
    const ws = parseFloat(windowSize) || 5;
    const sr = parseInt(sampleRate) || 32000;

    // Re-fetch fresh data to avoid stale state
    const [allAudio, allJobs] = await Promise.all([fetchAudioFiles(), fetchProcessingJobs()]);

    const folderFiles = allAudio.filter((af) => {
      const parent = af.folder_path ? af.folder_path.split("/")[0] : ROOT_SENTINEL;
      return parent === selectedFolder;
    });

    const unprocessed = folderFiles.filter((af) => {
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
      showMsg("warning", "All files in this folder are already processed for these settings");
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
    onModelUsed?.(mv);
  }, [selectedFolder, modelVersion, windowSize, sampleRate, defaultModel, onModelUsed]);

  const handleQueueAll = useCallback(async () => {
    const mv = modelVersion || defaultModel?.name || "";
    const ws = parseFloat(windowSize) || 5;
    const sr = parseInt(sampleRate) || 32000;

    const [allAudio, allJobs] = await Promise.all([fetchAudioFiles(), fetchProcessingJobs()]);

    // Scope to selected folder if one is picked
    const candidates = selectedFolder
      ? allAudio.filter((af) => {
          const parent = af.folder_path ? af.folder_path.split("/")[0] : ROOT_SENTINEL;
          return parent === selectedFolder;
        })
      : allAudio;

    const unprocessed = candidates.filter((af) => {
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
    onModelUsed?.(mv);
  }, [selectedFolder, modelVersion, windowSize, sampleRate, defaultModel, onModelUsed]);

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
            <label className="text-xs text-muted-foreground">Audio Files</label>
            <Select value={selectedFolder} onValueChange={setSelectedFolder}>
              <SelectTrigger className="h-9">
                <SelectValue placeholder="Select audio folder..." />
              </SelectTrigger>
              <SelectContent>
                {sortedFolders.map((key) => (
                  <SelectItem key={key} value={key}>
                    {folderDisplayName(key)} ({folderGroups.get(key)!.length} files)
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

        {folderStatus && folderStatus.unprocessed === 0 && (
          <p className="text-xs text-muted-foreground bg-muted px-3 py-2 rounded-md">
            All {folderStatus.total} files in this folder are already processed for the current settings.
          </p>
        )}
        {folderStatus && folderStatus.unprocessed > 0 && folderStatus.unprocessed < folderStatus.total && (
          <p className="text-xs text-muted-foreground">
            {folderStatus.unprocessed} of {folderStatus.total} files unprocessed
          </p>
        )}

        <div className="flex gap-2">
          <Button
            size="sm"
            onClick={handleQueueFolder}
            disabled={!selectedFolder || createJob.isPending || !!batchProgress}
          >
            Queue Folder
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

import { useState, useMemo } from "react";
import { ArrowLeft, ChevronLeft, ChevronRight } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { useSpectrogram, useEmbeddingSimilarity } from "@/hooks/queries/useAudioFiles";
import { useProcessingJobs } from "@/hooks/queries/useProcessing";
import { StatusBadge } from "@/components/shared/StatusBadge";
import { AudioPlayerBar } from "./AudioPlayerBar";
import { SpectrogramPlot } from "./SpectrogramPlot";
import { SimilarityMatrix } from "./SimilarityMatrix";
import { shortId, fmtDate } from "@/utils/format";
import type { AudioFile, EmbeddingSet } from "@/api/types";

interface AudioDetailProps {
  file: AudioFile;
  embeddingSets: EmbeddingSet[];
  onBack: () => void;
  onPrev?: () => void;
  onNext?: () => void;
}

export function AudioDetail({ file, embeddingSets, onBack, onPrev, onNext }: AudioDetailProps) {
  const [windowIndex, setWindowIndex] = useState(0);
  const [selectedEsId, setSelectedEsId] = useState<string | null>(
    embeddingSets.length > 0 ? embeddingSets[0].id : null,
  );

  const selectedEs = embeddingSets.find((es) => es.id === selectedEsId);
  const windowSize = selectedEs?.window_size_seconds ?? 5;
  const sampleRate = selectedEs?.target_sample_rate ?? 32000;

  const { data: spectrogram } = useSpectrogram(file.id, windowIndex, windowSize, sampleRate);
  const { data: similarity } = useEmbeddingSimilarity(file.id, selectedEsId);

  const totalWindows = spectrogram?.total_windows ?? (similarity?.num_windows ?? 0);

  const { data: allJobs = [] } = useProcessingJobs();
  const fileJobs = useMemo(
    () => allJobs.filter((j) => j.audio_file_id === file.id),
    [allJobs, file.id],
  );

  return (
    <div className="space-y-4">
      {/* Navigation bar */}
      <div className="flex items-center gap-2">
        <Button variant="ghost" size="sm" onClick={onBack}>
          <ArrowLeft className="h-4 w-4 mr-1" />
          Back
        </Button>
        <div className="flex-1" />
        {onPrev && (
          <Button variant="outline" size="sm" onClick={onPrev}>
            <ChevronLeft className="h-4 w-4 mr-1" />
            Prev
          </Button>
        )}
        {onNext && (
          <Button variant="outline" size="sm" onClick={onNext}>
            Next
            <ChevronRight className="h-4 w-4 ml-1" />
          </Button>
        )}
      </div>

      {/* File info */}
      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="text-base">
            {file.folder_path ? `${file.folder_path}/` : ""}
            {file.filename}
          </CardTitle>
        </CardHeader>
        <CardContent>
          <table className="text-sm w-full">
            <tbody>
              <tr>
                <td className="text-muted-foreground pr-4 py-0.5">ID</td>
                <td className="font-mono text-xs">{file.id}</td>
              </tr>
              <tr>
                <td className="text-muted-foreground pr-4 py-0.5">SHA-256</td>
                <td className="font-mono text-xs truncate max-w-[400px]">{file.checksum_sha256}</td>
              </tr>
              <tr>
                <td className="text-muted-foreground pr-4 py-0.5">Duration</td>
                <td>{file.duration_seconds != null ? `${file.duration_seconds.toFixed(1)}s` : "—"}</td>
              </tr>
              <tr>
                <td className="text-muted-foreground pr-4 py-0.5">Sample Rate</td>
                <td>{file.sample_rate_original != null ? `${file.sample_rate_original} Hz` : "—"}</td>
              </tr>
              <tr>
                <td className="text-muted-foreground pr-4 py-0.5">Created</td>
                <td>{fmtDate(file.created_at)}</td>
              </tr>
            </tbody>
          </table>
        </CardContent>
      </Card>

      {/* Processing history */}
      {fileJobs.length > 0 && (
        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-base">Processing History</CardTitle>
          </CardHeader>
          <CardContent>
            <table className="text-sm w-full">
              <thead>
                <tr className="border-b">
                  <th className="text-left py-1 font-medium">Job ID</th>
                  <th className="text-left py-1 font-medium">Status</th>
                  <th className="text-left py-1 font-medium">Model</th>
                  <th className="text-left py-1 font-medium">Window</th>
                  <th className="text-left py-1 font-medium">Date</th>
                </tr>
              </thead>
              <tbody>
                {fileJobs.map((job) => (
                  <tr key={job.id} className="border-b last:border-0">
                    <td className="py-1 font-mono text-xs">{shortId(job.id)}</td>
                    <td className="py-1">
                      <StatusBadge status={job.status} />
                      {job.warning_message && (
                        <span className="text-yellow-600 text-xs ml-2">{job.warning_message}</span>
                      )}
                      {job.error_message && (
                        <span className="text-red-600 text-xs ml-2">{job.error_message}</span>
                      )}
                    </td>
                    <td className="py-1">{job.model_version}</td>
                    <td className="py-1">{job.window_size_seconds}s</td>
                    <td className="py-1 text-xs text-muted-foreground">{fmtDate(job.created_at)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </CardContent>
        </Card>
      )}

      {/* Spectrogram & Embeddings visualization */}
      {embeddingSets.length > 0 && (
        <Card>
          <CardHeader className="pb-3">
            <div className="flex items-center gap-4 flex-wrap">
              <CardTitle className="text-base">Spectrogram & Embeddings</CardTitle>
              <Select value={selectedEsId ?? ""} onValueChange={setSelectedEsId}>
                <SelectTrigger className="w-[280px] h-8">
                  <SelectValue placeholder="Select embedding set" />
                </SelectTrigger>
                <SelectContent>
                  {embeddingSets.map((es) => (
                    <SelectItem key={es.id} value={es.id}>
                      {shortId(es.id)} — {es.model_version} ({es.vector_dim}d)
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
              <div className="flex items-center gap-2">
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => setWindowIndex((i) => Math.max(0, i - 1))}
                  disabled={windowIndex === 0}
                >
                  Prev
                </Button>
                <span className="text-sm text-muted-foreground">
                  Window {windowIndex} of {totalWindows}
                </span>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => setWindowIndex((i) => Math.min(totalWindows - 1, i + 1))}
                  disabled={windowIndex >= totalWindows - 1}
                >
                  Next
                </Button>
              </div>
            </div>
          </CardHeader>
          <CardContent className="space-y-4">
            <AudioPlayerBar
              audioId={file.id}
              totalWindows={totalWindows}
              windowSizeSeconds={windowSize}
              duration={file.duration_seconds ?? 0}
              activeWindow={windowIndex}
              onWindowClick={setWindowIndex}
            />
            {spectrogram && <SpectrogramPlot data={spectrogram} />}
            {similarity && (
              <SimilarityMatrix
                data={similarity}
                currentWindow={windowIndex}
                onWindowClick={setWindowIndex}
              />
            )}
          </CardContent>
        </Card>
      )}
    </div>
  );
}

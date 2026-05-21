import { useMemo } from "react";

import {
  MAX_EXPORT_WINDOW_SECONDS,
  type PianoRollMidiExportRead,
  type PianoRollMidiExportStatus,
  type PianoRollNotesStatus,
  isPianoRollMidiExportStatusAbsent,
  isPianoRollNotesStatusAbsent,
  pianoRollAudioExportDownloadUrl,
  pianoRollMidiExportDownloadUrl,
  useCreatePianoRollMidiExport,
  usePianoRollMidiExportStatus,
} from "@/api/sequenceModels";
import { Button } from "@/components/ui/button";

const WINDOW_MATCH_TOLERANCE_S = 0.05;

function notesAreComplete(status: PianoRollNotesStatus | undefined): boolean {
  if (!status) return false;
  if (isPianoRollNotesStatusAbsent(status)) return false;
  return status.status === "complete";
}

function exportStatusValue(
  status: PianoRollMidiExportStatus | undefined,
): "absent" | "queued" | "running" | "complete" | "failed" | "canceled" {
  if (!status || isPianoRollMidiExportStatusAbsent(status)) return "absent";
  return status.status;
}

function exportRow(
  status: PianoRollMidiExportStatus | undefined,
): PianoRollMidiExportRead | null {
  if (!status || isPianoRollMidiExportStatusAbsent(status)) return null;
  return status;
}

function windowsMatch(
  a: PianoRollMidiExportRead,
  startUtc: number,
  endUtc: number,
): boolean {
  return (
    Math.abs(a.window_start_utc - startUtc) <= WINDOW_MATCH_TOLERANCE_S &&
    Math.abs(a.window_end_utc - endUtc) <= WINDOW_MATCH_TOLERANCE_S
  );
}

function formatUtc(seconds: number): string {
  const ms = seconds * 1000;
  if (!Number.isFinite(ms)) return String(seconds);
  return new Date(ms).toISOString().replace("T", " ").replace(".000Z", "Z");
}

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
}

export function MidiExportButton({
  jobId,
  notesStatus,
  windowStartUtc,
  windowEndUtc,
}: {
  jobId: string;
  notesStatus: PianoRollNotesStatus | undefined;
  windowStartUtc: number;
  windowEndUtc: number;
}) {
  const { data: exportStatus } = usePianoRollMidiExportStatus(jobId);
  const mutation = useCreatePianoRollMidiExport(jobId);

  const notesReady = notesAreComplete(notesStatus);
  const value = exportStatusValue(exportStatus);
  const row = exportRow(exportStatus);

  const windowDurationS = windowEndUtc - windowStartUtc;
  const windowOverCap = windowDurationS > MAX_EXPORT_WINDOW_SECONDS;
  const windowInvalid =
    !Number.isFinite(windowDurationS) || windowDurationS <= 0;
  const windowDisabledReason = useMemo(() => {
    if (windowInvalid) return "Pan or zoom to a positive-duration window.";
    if (windowOverCap) {
      return `Export window exceeds the ${MAX_EXPORT_WINDOW_SECONDS / 60}-minute cap.`;
    }
    return null;
  }, [windowInvalid, windowOverCap]);

  const matchesPersistedWindow =
    row != null && row.status === "complete"
      ? windowsMatch(row, windowStartUtc, windowEndUtc)
      : false;

  const inFlight = value === "queued" || value === "running";
  const isFailed = value === "failed";
  const failedMessage = row && isFailed ? row.error_message : null;

  const submit = (force: boolean) => {
    if (!notesReady || windowDisabledReason !== null || mutation.isPending) {
      return;
    }
    mutation.mutate({
      window_start_utc: windowStartUtc,
      window_end_utc: windowEndUtc,
      ...(force ? { force: true } : {}),
    });
  };

  // ---- render ----
  if (!notesReady) {
    return (
      <Button
        type="button"
        variant="outline"
        size="sm"
        className="h-8 border-zinc-700 bg-zinc-900 px-2 text-zinc-100 disabled:cursor-not-allowed disabled:opacity-60"
        disabled
        title="Notes must finish before MIDI can be exported."
        data-testid="eej-piano-roll-midi-export-button"
        data-status="absent"
      >
        Export view
      </Button>
    );
  }

  if (inFlight) {
    return (
      <span
        className="inline-flex h-8 items-center rounded border border-zinc-700 bg-zinc-900 px-2 text-xs text-zinc-300"
        data-testid="eej-piano-roll-midi-export-progress"
        data-status={value}
      >
        Exporting MIDI and audio…
      </span>
    );
  }

  if (row != null && row.status === "complete") {
    const reExportTitle = windowDisabledReason
      ? windowDisabledReason
      : matchesPersistedWindow
        ? "Current view matches the exported window."
        : "Current view differs from the exported window.";
    return (
      <div
        className="flex flex-wrap items-center gap-2"
        data-testid="eej-piano-roll-midi-export"
        data-status="complete"
      >
        <span
          className="text-[11px] text-zinc-400"
          data-testid="eej-piano-roll-midi-export-window"
        >
          Exported window: {formatUtc(row.window_start_utc)} →{" "}
          {formatUtc(row.window_end_utc)} (
          {(row.window_end_utc - row.window_start_utc).toFixed(1)} s)
        </span>
        <a
          className="inline-flex h-8 items-center rounded border border-zinc-700 bg-zinc-900 px-2 text-xs text-zinc-100 hover:bg-zinc-800"
          href={pianoRollMidiExportDownloadUrl(jobId)}
          download
          data-testid="eej-piano-roll-midi-export-download"
        >
          Download MIDI{row.n_bytes != null ? ` (${formatBytes(row.n_bytes)})` : ""}
        </a>
        <a
          className="inline-flex h-8 items-center rounded border border-zinc-700 bg-zinc-900 px-2 text-xs text-zinc-100 hover:bg-zinc-800"
          href={pianoRollAudioExportDownloadUrl(jobId)}
          download
          data-testid="eej-piano-roll-audio-export-download"
        >
          Download audio (FLAC) ({formatBytes(row.audio_size_bytes)})
        </a>
        <Button
          type="button"
          variant="outline"
          size="sm"
          className={
            matchesPersistedWindow
              ? "h-8 border-zinc-700 bg-zinc-900 px-2 text-xs text-zinc-300 hover:bg-zinc-800"
              : "h-8 border-emerald-700 bg-emerald-950 px-2 text-xs text-emerald-100 hover:bg-emerald-900"
          }
          disabled={windowDisabledReason !== null || mutation.isPending}
          title={reExportTitle}
          onClick={() => submit(true)}
          data-testid="eej-piano-roll-midi-export-button"
          data-status="complete"
          data-window-match={matchesPersistedWindow ? "true" : "false"}
        >
          Re-export view
        </Button>
      </div>
    );
  }

  // Absent OR failed OR canceled
  return (
    <div
      className="flex items-center gap-2"
      data-testid="eej-piano-roll-midi-export"
      data-status={value}
    >
      <Button
        type="button"
        variant="outline"
        size="sm"
        className="h-8 border-zinc-700 bg-zinc-900 px-2 text-zinc-100 hover:bg-zinc-800 disabled:cursor-not-allowed disabled:opacity-60"
        disabled={windowDisabledReason !== null || mutation.isPending}
        title={windowDisabledReason ?? (failedMessage || undefined) ?? undefined}
        onClick={() => submit(false)}
        data-testid="eej-piano-roll-midi-export-button"
        data-status={value}
      >
        {isFailed ? "Retry export" : "Export view"}
      </Button>
      {isFailed && failedMessage ? (
        <span
          className="max-w-xs truncate rounded border border-red-700 bg-red-950 px-2 py-1 text-[11px] text-red-200"
          title={failedMessage}
          data-testid="eej-piano-roll-midi-export-error"
        >
          {failedMessage}
        </span>
      ) : null}
    </div>
  );
}

import { useState, useRef, useEffect } from "react";

import {
  type PianoRollMidiExportStatus,
  type PianoRollNotesStatus,
  isPianoRollMidiExportStatusAbsent,
  isPianoRollNotesStatusAbsent,
  pianoRollMidiExportDownloadUrl,
  useCreatePianoRollMidiExport,
  usePianoRollMidiExportStatus,
} from "@/api/sequenceModels";
import { Button } from "@/components/ui/button";

type ButtonState = {
  label: string;
  disabled: boolean;
  title: string | null;
  variant: "primary" | "download";
};

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

function deriveButtonState(
  notesStatus: PianoRollNotesStatus | undefined,
  exportStatus: PianoRollMidiExportStatus | undefined,
): ButtonState {
  if (!notesAreComplete(notesStatus)) {
    return {
      label: "Export MIDI",
      disabled: true,
      title: "Notes must finish before MIDI can be exported.",
      variant: "primary",
    };
  }
  const value = exportStatusValue(exportStatus);
  if (value === "queued" || value === "running") {
    return {
      label: "Exporting…",
      disabled: true,
      title: "MIDI export is running.",
      variant: "primary",
    };
  }
  if (value === "complete") {
    return {
      label: "Download MIDI",
      disabled: false,
      title: null,
      variant: "download",
    };
  }
  return {
    label: "Export MIDI",
    disabled: false,
    title: null,
    variant: "primary",
  };
}

function downloadHref(jobId: string): string {
  return pianoRollMidiExportDownloadUrl(jobId);
}

export function MidiExportButton({
  jobId,
  notesStatus,
}: {
  jobId: string;
  notesStatus: PianoRollNotesStatus | undefined;
}) {
  const { data: exportStatus } = usePianoRollMidiExportStatus(jobId);
  const mutation = useCreatePianoRollMidiExport(jobId);
  const [menuOpen, setMenuOpen] = useState(false);
  const menuRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    if (!menuOpen) return;
    const handleClick = (event: MouseEvent) => {
      if (!menuRef.current) return;
      if (!menuRef.current.contains(event.target as Node)) {
        setMenuOpen(false);
      }
    };
    document.addEventListener("mousedown", handleClick);
    return () => document.removeEventListener("mousedown", handleClick);
  }, [menuOpen]);

  const state = deriveButtonState(notesStatus, exportStatus);
  const value = exportStatusValue(exportStatus);
  const failedMessage =
    exportStatus && !isPianoRollMidiExportStatusAbsent(exportStatus)
      ? exportStatus.error_message
      : null;
  const isFailed = value === "failed";
  const tooltip = isFailed && failedMessage ? failedMessage : state.title;

  const onPrimary = () => {
    if (state.disabled) return;
    if (state.variant === "download") {
      window.location.assign(downloadHref(jobId));
      return;
    }
    mutation.mutate({});
  };

  const onReExport = () => {
    setMenuOpen(false);
    mutation.mutate({ force: true });
  };

  return (
    <div
      className="relative flex items-center gap-1"
      data-testid="eej-piano-roll-midi-export"
      ref={menuRef}
    >
      <Button
        type="button"
        variant="outline"
        size="sm"
        className="h-8 border-zinc-700 bg-zinc-900 px-2 text-zinc-100 hover:bg-zinc-800 disabled:cursor-not-allowed disabled:opacity-60"
        disabled={state.disabled || mutation.isPending}
        onClick={onPrimary}
        title={tooltip ?? undefined}
        data-testid="eej-piano-roll-midi-export-button"
        data-status={value}
      >
        {state.label}
      </Button>
      {state.variant === "download" ? (
        <>
          <Button
            type="button"
            variant="outline"
            size="sm"
            className="h-8 w-6 border-zinc-700 bg-zinc-900 px-0 text-zinc-100 hover:bg-zinc-800"
            onClick={() => setMenuOpen((open) => !open)}
            aria-label="MIDI export options"
            data-testid="eej-piano-roll-midi-export-menu-button"
          >
            ⋮
          </Button>
          {menuOpen ? (
            <div
              className="absolute right-0 top-full z-10 mt-1 w-32 rounded border border-zinc-700 bg-zinc-900 text-xs text-zinc-100 shadow-lg"
              data-testid="eej-piano-roll-midi-export-menu"
            >
              <button
                type="button"
                className="block w-full px-3 py-2 text-left hover:bg-zinc-800"
                onClick={onReExport}
                data-testid="eej-piano-roll-midi-export-rerun"
              >
                Re-export
              </button>
            </div>
          ) : null}
        </>
      ) : null}
    </div>
  );
}

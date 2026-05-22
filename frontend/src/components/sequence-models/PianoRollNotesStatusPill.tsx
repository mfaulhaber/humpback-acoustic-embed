import { cn } from "@/lib/utils";
import {
  isPianoRollNotesStatusAbsent,
  type PianoRollNotesJobStatus,
  type PianoRollNotesStatus,
} from "@/api/sequenceModels";

const STATUS_LABEL: Record<PianoRollNotesJobStatus | "absent", string> = {
  absent: "absent",
  queued: "queued",
  running: "running",
  complete: "complete",
  failed: "failed",
  canceled: "canceled",
};

const STATUS_CLASS: Record<PianoRollNotesJobStatus | "absent", string> = {
  absent: "bg-zinc-100 text-zinc-600 border-zinc-200",
  queued: "bg-blue-100 text-blue-800 border-blue-200",
  running: "bg-yellow-100 text-yellow-800 border-yellow-200",
  complete: "bg-green-100 text-green-800 border-green-200",
  failed: "bg-red-100 text-red-800 border-red-200",
  canceled: "bg-gray-100 text-gray-600 border-gray-200",
};

interface PianoRollNotesStatusPillProps {
  status: PianoRollNotesStatus;
  className?: string;
  prefix?: string;
  /**
   * Optional click handler that enqueues a v3 notes job. Surfaced as a
   * "v3 available" badge next to the main pill when the most-recent
   * complete row is older than v3 (per ADR-069 §9.5).
   */
  onRequestV3Upgrade?: () => void;
}

export function PianoRollNotesStatusPill({
  status,
  className,
  prefix = "Notes",
  onRequestV3Upgrade,
}: PianoRollNotesStatusPillProps) {
  const value = status.status;
  const showV3Badge =
    onRequestV3Upgrade != null &&
    !isPianoRollNotesStatusAbsent(status) &&
    status.status === "complete" &&
    status.extractor_version < "v3";
  return (
    <span
      data-testid="piano-roll-notes-status-pill"
      data-notes-status={value}
      className={cn(
        "inline-flex items-center gap-2 rounded-full border px-2 py-0.5 text-xs font-medium",
        STATUS_CLASS[value],
        className,
      )}
    >
      <span>{prefix}: {STATUS_LABEL[value]}</span>
      {showV3Badge ? (
        <button
          type="button"
          onClick={onRequestV3Upgrade}
          className="rounded-full bg-emerald-600 px-2 py-0.5 text-[10px] font-semibold text-white hover:bg-emerald-500"
          data-testid="piano-roll-notes-v3-upgrade-badge"
          title="Re-extract Piano Roll Notes at v3 to enable MPE export and ribbon rendering."
        >
          v3 available
        </button>
      ) : null}
    </span>
  );
}

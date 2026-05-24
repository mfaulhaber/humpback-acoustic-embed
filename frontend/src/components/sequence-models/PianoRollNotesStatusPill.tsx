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
   * Optional click handler that enqueues an upgrade notes job at
   * ``latestExtractorVersion``. Surfaced as an "<X> available" badge
   * next to the main pill when the most-recent complete row is older
   * than ``latestExtractorVersion`` (lexicographic comparison; the
   * backend resolver uses the same ordering).
   */
  onRequestUpgrade?: () => void;
  /**
   * The current default extractor version. The upgrade badge appears
   * when the displayed status is a complete row at a version
   * lexicographically less than this. Required for the badge to render.
   * Mirrors ``humpback.models.piano_roll_notes.DEFAULT_EXTRACTOR_VERSION``.
   */
  latestExtractorVersion?: string;
}

export function PianoRollNotesStatusPill({
  status,
  className,
  prefix = "Notes",
  onRequestUpgrade,
  latestExtractorVersion,
}: PianoRollNotesStatusPillProps) {
  const value = status.status;
  const showUpgradeBadge =
    onRequestUpgrade != null &&
    latestExtractorVersion != null &&
    !isPianoRollNotesStatusAbsent(status) &&
    status.status === "complete" &&
    status.extractor_version < latestExtractorVersion;
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
      {showUpgradeBadge ? (
        <button
          type="button"
          onClick={onRequestUpgrade}
          className="rounded-full bg-emerald-600 px-2 py-0.5 text-[10px] font-semibold text-white hover:bg-emerald-500"
          data-testid="piano-roll-notes-upgrade-badge"
          title={`Re-extract Piano Roll Notes at ${latestExtractorVersion}.`}
        >
          {latestExtractorVersion} available
        </button>
      ) : null}
    </span>
  );
}

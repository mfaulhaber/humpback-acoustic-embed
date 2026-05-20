import { cn } from "@/lib/utils";
import type {
  PianoRollNotesJobStatus,
  PianoRollNotesStatus,
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
}

export function PianoRollNotesStatusPill({
  status,
  className,
  prefix = "Notes",
}: PianoRollNotesStatusPillProps) {
  const value = status.status;
  return (
    <span
      data-testid="piano-roll-notes-status-pill"
      data-notes-status={value}
      className={cn(
        "inline-flex items-center rounded-full border px-2 py-0.5 text-xs font-medium",
        STATUS_CLASS[value],
        className,
      )}
    >
      {prefix}: {STATUS_LABEL[value]}
    </span>
  );
}

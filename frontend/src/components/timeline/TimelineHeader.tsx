// frontend/src/components/timeline/TimelineHeader.tsx
import { useNavigate } from "react-router-dom";
import { ArrowLeft, RefreshCw, Loader2, CheckCircle } from "lucide-react";
import { COLORS } from "./constants";

interface TimelineHeaderProps {
  hydrophone: string;
  startTimestamp: number;
  endTimestamp: number;
  syncNeeded?: boolean | null;
  isSyncing?: boolean;
  syncSummary?: string | null;
  onSyncEmbeddings?: () => void;
}

export function TimelineHeader({
  hydrophone,
  startTimestamp,
  endTimestamp,
  syncNeeded,
  isSyncing,
  syncSummary,
  onSyncEmbeddings,
}: TimelineHeaderProps) {
  const navigate = useNavigate();
  const startStr = new Date(startTimestamp * 1000).toISOString().replace("T", " ").slice(0, 19) + "Z";
  const endStr = new Date(endTimestamp * 1000).toISOString().replace("T", " ").slice(0, 19) + "Z";

  // Parse sync summary for display
  let summaryText: string | null = null;
  if (syncSummary) {
    try {
      const s = JSON.parse(syncSummary);
      const parts: string[] = [];
      if (s.added > 0) parts.push(`${s.added} added`);
      if (s.removed > 0) parts.push(`${s.removed} removed`);
      if (s.skipped > 0) parts.push(`${s.skipped} skipped`);
      summaryText = parts.length > 0 ? parts.join(", ") : "already in sync";
    } catch {
      /* ignore parse errors */
    }
  }

  return (
    <div
      className="flex items-center justify-between px-4 py-2 shrink-0"
      style={{ background: COLORS.headerBg, borderBottom: `1px solid ${COLORS.border}` }}
    >
      <div className="flex items-center gap-3">
        <button
          onClick={() => navigate("/app/classifier/hydrophone")}
          className="flex items-center gap-1 text-xs hover:opacity-80"
          style={{ color: COLORS.textMuted }}
        >
          <ArrowLeft size={14} /> Back to Jobs
        </button>
        <span className="font-bold text-sm" style={{ color: COLORS.accent }}>
          {hydrophone}
        </span>
        <span className="text-xs" style={{ color: COLORS.textBright }}>
          {startStr} — {endStr}
        </span>
      </div>

      {/* Sync embeddings button */}
      <div className="flex items-center gap-2">
        {summaryText && !isSyncing && (
          <span className="flex items-center gap-1 text-xs" style={{ color: COLORS.textMuted }}>
            <CheckCircle size={12} style={{ color: "#22c55e" }} />
            {summaryText}
          </span>
        )}
        {syncNeeded && onSyncEmbeddings && (
          <button
            onClick={onSyncEmbeddings}
            disabled={isSyncing}
            className="flex items-center gap-1 text-xs px-2 py-1 rounded hover:opacity-80"
            style={{
              background: isSyncing ? "transparent" : COLORS.accent,
              color: isSyncing ? COLORS.textMuted : COLORS.bg,
              border: `1px solid ${isSyncing ? COLORS.border : COLORS.accent}`,
              cursor: isSyncing ? "not-allowed" : "pointer",
            }}
          >
            {isSyncing ? (
              <Loader2 size={12} className="animate-spin" />
            ) : (
              <RefreshCw size={12} />
            )}
            {isSyncing ? "Syncing..." : "Sync Embeddings"}
          </button>
        )}
      </div>
    </div>
  );
}

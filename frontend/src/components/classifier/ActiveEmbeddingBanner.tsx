import { Link } from "react-router-dom";
import { Loader2 } from "lucide-react";
import { useEmbeddingJobs } from "@/hooks/queries/useVocalization";

export function ActiveEmbeddingBanner() {
  const { data: jobs = [] } = useEmbeddingJobs(0, 10);
  const active = jobs.filter(
    (j) => j.status === "queued" || j.status === "running",
  );

  if (active.length === 0) return null;

  const running = active.filter((j) => j.status === "running");
  const queued = active.filter((j) => j.status === "queued");

  return (
    <div className="flex items-center gap-2 rounded-md border border-blue-200 bg-blue-50 px-3 py-2 text-sm text-blue-800">
      <Loader2 className="h-4 w-4 animate-spin shrink-0" />
      <span>
        {running.length > 0 && (
          <>
            {running.length} embedding {running.length === 1 ? "job" : "jobs"} running
            {running.length === 1 &&
              running[0].progress_total != null &&
              running[0].progress_total > 0 && (
                <span className="text-blue-600 ml-1">
                  ({running[0].progress_current ?? 0}/{running[0].progress_total})
                </span>
              )}
          </>
        )}
        {running.length > 0 && queued.length > 0 && ", "}
        {queued.length > 0 && (
          <>
            {queued.length} queued
          </>
        )}
      </span>
      <Link
        to="/app/classifier/embeddings"
        className="ml-auto text-xs text-blue-600 hover:underline shrink-0"
      >
        View details
      </Link>
    </div>
  );
}

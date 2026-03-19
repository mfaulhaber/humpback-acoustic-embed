import { useRef, useState } from "react";
import { AlertCircle, X } from "lucide-react";
import { useHealth } from "@/hooks/queries/useHealth";

export function DatabaseErrorBanner() {
  const { data } = useHealth();
  const [dismissed, setDismissed] = useState(false);
  const prevStatusRef = useRef<string | undefined>(undefined);

  // Re-show banner if health status changes to an error after being dismissed
  if (data?.status !== prevStatusRef.current) {
    prevStatusRef.current = data?.status;
    if (data?.status !== "ok") {
      setDismissed(false);
    }
  }

  if (!data || data.status === "ok" || dismissed) {
    return null;
  }

  const title =
    data.status === "starting" ? "Database connecting…" : "Database unavailable";

  return (
    <div className="w-full bg-red-50 border-b border-red-300 text-red-800 px-4 py-2 flex items-start gap-2 text-sm">
      <AlertCircle className="h-4 w-4 mt-0.5 shrink-0" />
      <span className="font-medium">{title}</span>
      {data.detail && <span className="flex-1">{data.detail}</span>}
      <button
        onClick={() => setDismissed(true)}
        className="ml-auto shrink-0 hover:text-red-600"
        aria-label="Dismiss"
      >
        <X className="h-4 w-4" />
      </button>
    </div>
  );
}

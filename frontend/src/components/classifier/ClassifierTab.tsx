import { useState } from "react";
import { cn } from "@/lib/utils";
import { TrainingTab } from "./TrainingTab";
import { DetectionTab } from "./DetectionTab";

type SubView = "train" | "detect";

export function ClassifierTab() {
  const [view, setView] = useState<SubView>("train");

  return (
    <div className="space-y-4">
      <div className="flex gap-2 border-b pb-2">
        <button
          className={cn(
            "px-4 py-1.5 text-sm font-medium rounded-t transition-colors",
            view === "train"
              ? "bg-slate-800 text-white"
              : "text-muted-foreground hover:text-foreground"
          )}
          onClick={() => setView("train")}
        >
          Train
        </button>
        <button
          className={cn(
            "px-4 py-1.5 text-sm font-medium rounded-t transition-colors",
            view === "detect"
              ? "bg-slate-800 text-white"
              : "text-muted-foreground hover:text-foreground"
          )}
          onClick={() => setView("detect")}
        >
          Detect
        </button>
      </div>

      {view === "train" ? <TrainingTab /> : <DetectionTab />}
    </div>
  );
}

import { useEffect } from "react";
import { ChevronLeft, ChevronRight } from "lucide-react";

export interface RegionInfo {
  regionId: string;
  startTimestamp: number;
  endTimestamp: number;
}

export interface RegionNavBarProps {
  regions: RegionInfo[];
  activeIndex: number;
  onPrev: () => void;
  onNext: () => void;
  /**
   * When ``true``, A/D keyboard shortcuts trigger ``onPrev`` / ``onNext``
   * unless focus is in an input/textarea/contenteditable element.
   */
  enableKeyboardShortcuts?: boolean;
  testId?: string;
}

function isInputFocused(): boolean {
  const el = document.activeElement;
  if (!el) return false;
  const tag = (el as HTMLElement).tagName;
  if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT") return true;
  const editable = (el as HTMLElement).isContentEditable;
  return Boolean(editable);
}

export function RegionNavBar({
  regions,
  activeIndex,
  onPrev,
  onNext,
  enableKeyboardShortcuts = true,
  testId,
}: RegionNavBarProps) {
  useEffect(() => {
    if (!enableKeyboardShortcuts) return;
    const handler = (e: KeyboardEvent) => {
      if (e.metaKey || e.ctrlKey || e.altKey) return;
      if (isInputFocused()) return;
      if (e.key === "a" || e.key === "A") {
        e.preventDefault();
        onPrev();
      } else if (e.key === "d" || e.key === "D") {
        e.preventDefault();
        onNext();
      }
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [enableKeyboardShortcuts, onPrev, onNext]);

  if (regions.length === 0) return null;
  const region = regions[activeIndex];

  return (
    <div
      className="flex items-center gap-1 text-sm"
      data-testid={testId ?? "region-nav-bar"}
    >
      <button
        className="rounded-md border p-1 hover:bg-accent disabled:opacity-30"
        disabled={activeIndex === 0}
        onClick={onPrev}
        title="Previous region (A)"
        data-testid="region-nav-prev"
      >
        <ChevronLeft className="h-4 w-4" />
      </button>
      <button
        className="rounded-md border p-1 hover:bg-accent disabled:opacity-30"
        disabled={activeIndex === regions.length - 1}
        onClick={onNext}
        title="Next region (D)"
        data-testid="region-nav-next"
      >
        <ChevronRight className="h-4 w-4" />
      </button>
      <span className="text-muted-foreground text-xs">
        Region {activeIndex + 1}/{regions.length}
        {region ? ` · ${region.regionId.slice(0, 8)}` : ""}
      </span>
    </div>
  );
}

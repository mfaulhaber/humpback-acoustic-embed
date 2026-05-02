// HMMStateBar is a thin shim over the source-agnostic
// :class:`DiscreteSequenceBar` (ADR-061). Existing imports continue to
// work with the same prop shape; new code should import
// ``DiscreteSequenceBar`` directly.

import {
  DiscreteSequenceBar,
  type DiscreteSequenceItem,
  visibleItems,
  regionBoundaryXPositions,
} from "./DiscreteSequenceBar";

export interface ViterbiWindow {
  start_timestamp: number;
  end_timestamp: number;
  viterbi_state: number;
  max_state_probability: number;
}

interface HMMStateBarProps {
  items: ViterbiWindow[];
  nStates: number;
  currentRegion?: {
    startTimestamp: number;
    endTimestamp: number;
  } | null;
}

export function HMMStateBar({ items, nStates, currentRegion }: HMMStateBarProps) {
  const adapted: DiscreteSequenceItem[] = items.map((w) => ({
    start_timestamp: w.start_timestamp,
    end_timestamp: w.end_timestamp,
    label: w.viterbi_state,
    confidence: w.max_state_probability,
  }));
  return (
    <DiscreteSequenceBar
      items={adapted}
      numLabels={nStates}
      mode="rows"
      currentRegion={currentRegion}
      testId="hmm-state-bar"
      ariaLabel="HMM state timeline"
      tooltipFormatter={(item) =>
        `State ${item.label} · ${item.start_timestamp.toFixed(1)}s–${item.end_timestamp.toFixed(1)}s · prob ${(item.confidence ?? 0).toFixed(3)}`
      }
    />
  );
}

// Backwards-compat exports for tests that import the legacy helpers.
function visibleWindows(items: ViterbiWindow[], viewStart: number, viewEnd: number) {
  return items.filter(
    (item) => item.end_timestamp >= viewStart && item.start_timestamp <= viewEnd,
  );
}

export { visibleWindows, regionBoundaryXPositions };
// Keep the un-renamed export so non-test callers continue to compile if any
// remain. ``visibleItems`` is the canonical name.
export { visibleItems };

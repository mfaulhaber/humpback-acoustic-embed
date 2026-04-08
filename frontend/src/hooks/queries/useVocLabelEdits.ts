import { useReducer, useMemo } from "react";
import type {
  TimelineVocalizationLabel,
  VocalizationLabelBatchEditItem,
} from "@/api/types";

// Per-row pending edits: labels to add and labels to remove
interface RowEdits {
  adds: Set<string>;
  removes: Set<string>;
}

interface State {
  selectedRowId: string | null;
  edits: Map<string, RowEdits>;
}

export type VocLabelAction =
  | { type: "toggle_add"; row_id: string; label: string }
  | { type: "toggle_remove"; row_id: string; label: string }
  | { type: "select"; row_id: string }
  | { type: "deselect" }
  | { type: "clear" };

const initialState: State = {
  selectedRowId: null,
  edits: new Map(),
};

function getOrCreateRowEdits(edits: Map<string, RowEdits>, rowId: string): RowEdits {
  const existing = edits.get(rowId);
  if (existing) return { adds: new Set(existing.adds), removes: new Set(existing.removes) };
  return { adds: new Set(), removes: new Set() };
}

function cleanupRow(edits: Map<string, RowEdits>, rowId: string) {
  const row = edits.get(rowId);
  if (row && row.adds.size === 0 && row.removes.size === 0) {
    edits.delete(rowId);
  }
}

function reducer(state: State, action: VocLabelAction): State {
  switch (action.type) {
    case "toggle_add": {
      const newEdits = new Map(state.edits);
      const row = getOrCreateRowEdits(newEdits, action.row_id);

      if (row.adds.has(action.label)) {
        // Un-add: toggle off a pending add
        row.adds.delete(action.label);
      } else {
        // Add the label
        row.adds.add(action.label);
        // If it was pending removal, cancel that
        row.removes.delete(action.label);

        // Mutual exclusivity: "(Negative)" vs type labels
        if (action.label === "(Negative)") {
          // Adding (Negative) clears all type adds and queues type removes
          // We don't know saved labels here — removes for saved types are
          // handled via toggle_remove by the UI, but we clear pending type adds.
          for (const lbl of row.adds) {
            if (lbl !== "(Negative)") row.adds.delete(lbl);
          }
        } else {
          // Adding a type clears (Negative) add
          row.adds.delete("(Negative)");
        }
      }

      newEdits.set(action.row_id, row);
      cleanupRow(newEdits, action.row_id);
      return { ...state, edits: newEdits };
    }

    case "toggle_remove": {
      const newEdits = new Map(state.edits);
      const row = getOrCreateRowEdits(newEdits, action.row_id);

      if (row.removes.has(action.label)) {
        // Un-remove: cancel a pending removal
        row.removes.delete(action.label);
      } else {
        // Mark for removal
        row.removes.add(action.label);
        // If it was pending add, cancel that instead
        row.adds.delete(action.label);
      }

      newEdits.set(action.row_id, row);
      cleanupRow(newEdits, action.row_id);
      return { ...state, edits: newEdits };
    }

    case "select":
      return { ...state, selectedRowId: action.row_id };

    case "deselect":
      return { ...state, selectedRowId: null };

    case "clear":
      return initialState;

    default:
      return state;
  }
}

export type LabelDisplayState = "saved" | "inference" | "pending_add" | "pending_remove";

export interface SavedLabel {
  label: string;
  source: "manual" | "inference";
}

/**
 * Compute the effective labels for a row given saved labels (manual + inference)
 * and pending edits.
 */
export function computeEffectiveLabels(
  savedLabels: SavedLabel[],
  rowEdits: RowEdits | undefined,
): { label: string; state: LabelDisplayState }[] {
  const result: { label: string; state: LabelDisplayState }[] = [];

  for (const { label, source } of savedLabels) {
    if (source === "manual") {
      if (rowEdits?.removes.has(label)) {
        result.push({ label, state: "pending_remove" });
      } else {
        result.push({ label, state: "saved" });
      }
    } else {
      // Inference labels: if the user has added a manual version, show as saved
      // instead of inference (promotion). Otherwise show as inference.
      if (rowEdits?.adds.has(label)) {
        result.push({ label, state: "saved" });
      } else {
        result.push({ label, state: "inference" });
      }
    }
  }

  // Pending adds: labels not already in saved set (manual or inference)
  if (rowEdits) {
    const existingLabels = new Set(savedLabels.map((sl) => sl.label));
    for (const label of rowEdits.adds) {
      if (!existingLabels.has(label)) {
        result.push({ label, state: "pending_add" });
      }
    }
  }

  return result;
}

/**
 * Serialize the edits map into batch API edit items.
 */
export function serializeEdits(
  edits: Map<string, RowEdits>,
): VocalizationLabelBatchEditItem[] {
  const items: VocalizationLabelBatchEditItem[] = [];

  for (const [rowId, rowEdits] of edits) {
    for (const label of rowEdits.removes) {
      items.push({ action: "delete", row_id: rowId, label });
    }
    for (const label of rowEdits.adds) {
      items.push({ action: "add", row_id: rowId, label, source: "manual" });
    }
  }

  return items;
}

/**
 * Build a map of row_id -> saved labels (manual + inference) from timeline vocalization labels.
 */
export function buildSavedLabelMap(
  labels: TimelineVocalizationLabel[],
  rowIdByUtc: Map<string, string>,
): Map<string, SavedLabel[]> {
  const map = new Map<string, SavedLabel[]>();

  for (const lbl of labels) {
    const key = `${lbl.start_utc}_${lbl.end_utc}`;
    const rowId = rowIdByUtc.get(key);
    if (!rowId) continue;
    const existing = map.get(rowId) ?? [];
    existing.push({
      label: lbl.label,
      source: lbl.source === "manual" ? "manual" : "inference",
    });
    map.set(rowId, existing);
  }

  return map;
}

export function useVocLabelEdits() {
  const [state, dispatch] = useReducer(reducer, initialState);

  const isDirty = useMemo(() => {
    for (const row of state.edits.values()) {
      if (row.adds.size > 0 || row.removes.size > 0) return true;
    }
    return false;
  }, [state.edits]);

  const editCount = useMemo(() => {
    let count = 0;
    for (const row of state.edits.values()) {
      count += row.adds.size + row.removes.size;
    }
    return count;
  }, [state.edits]);

  return {
    state,
    dispatch,
    isDirty,
    editCount,
    selectedRowId: state.selectedRowId,
  };
}

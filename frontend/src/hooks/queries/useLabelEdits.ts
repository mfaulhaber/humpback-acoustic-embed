import { useReducer, useMemo } from "react";
import { v4 as uuidv4 } from "uuid";
import type { DetectionRow } from "@/api/types";
import type { LabelType } from "@/components/timeline/constants";

export type { LabelType };

export interface LabelEdit {
  action: "add" | "move" | "delete" | "change_type" | "clear_label";
  id?: string;
  row_id?: string;
  start_utc?: number;
  end_utc?: number;
  label?: LabelType;
}

interface State {
  edits: LabelEdit[];
  selectedId: string | null;
}

export type Action =
  | { type: "add"; start_utc: number; end_utc: number; label: LabelType }
  | { type: "move"; row_id: string; start_utc: number; end_utc: number }
  | { type: "delete"; row_id: string }
  | { type: "delete_by_id"; id: string }
  | { type: "change_type"; row_id: string; label: LabelType }
  | { type: "clear_label"; row_id: string }
  | { type: "select"; id: string | null }
  | { type: "clear" };

const initialState: State = {
  edits: [],
  selectedId: null,
};

function reducer(state: State, action: Action): State {
  switch (action.type) {
    case "add": {
      const newId = uuidv4();
      const newEdit: LabelEdit = {
        action: "add",
        id: newId,
        start_utc: action.start_utc,
        end_utc: action.end_utc,
        label: action.label,
      };
      return {
        edits: [...state.edits, newEdit],
        selectedId: newId,
      };
    }

    case "move": {
      // Collapse with an existing move or add edit for the same row
      const existingIdx = state.edits.findIndex(
        (e) =>
          (e.action === "move" && e.row_id === action.row_id) ||
          (e.action === "add" && e.id === action.row_id),
      );

      if (existingIdx !== -1) {
        const existing = state.edits[existingIdx];
        const updated: LabelEdit = {
          ...existing,
          start_utc: action.start_utc,
          end_utc: action.end_utc,
        };
        const edits = [...state.edits];
        edits[existingIdx] = updated;
        return { ...state, edits };
      }

      // No existing move/add — create a new move edit
      const newEdit: LabelEdit = {
        action: "move",
        row_id: action.row_id,
        start_utc: action.start_utc,
        end_utc: action.end_utc,
      };
      return { ...state, edits: [...state.edits, newEdit] };
    }

    case "delete": {
      // If deleting an "add" edit, just remove it from the list
      const addIdx = state.edits.findIndex(
        (e) => e.action === "add" && e.id === action.row_id,
      );
      if (addIdx !== -1) {
        const editId = state.edits[addIdx].id;
        const edits = state.edits.filter((_, i) => i !== addIdx);
        return {
          edits,
          selectedId: state.selectedId === editId ? null : state.selectedId,
        };
      }

      // Otherwise remove all prior edits for this row and add a delete edit
      const edits = state.edits.filter((e) => e.row_id !== action.row_id);
      const deleteEdit: LabelEdit = {
        action: "delete",
        row_id: action.row_id,
      };
      return {
        edits: [...edits, deleteEdit],
        selectedId:
          state.selectedId === action.row_id ? null : state.selectedId,
      };
    }

    case "delete_by_id": {
      // Delete an "add" edit by its generated id
      const addIdx = state.edits.findIndex(
        (e) => e.action === "add" && e.id === action.id,
      );
      if (addIdx !== -1) {
        const edits = state.edits.filter((_, i) => i !== addIdx);
        return {
          edits,
          selectedId:
            state.selectedId === action.id ? null : state.selectedId,
        };
      }
      return state;
    }

    case "change_type": {
      // Collapse with existing add or change_type for the same row
      const existingIdx = state.edits.findIndex(
        (e) =>
          (e.action === "add" && e.id === action.row_id) ||
          (e.action === "change_type" && e.row_id === action.row_id),
      );

      if (existingIdx !== -1) {
        const updated: LabelEdit = {
          ...state.edits[existingIdx],
          label: action.label,
        };
        const edits = [...state.edits];
        edits[existingIdx] = updated;
        return { ...state, edits };
      }

      // No existing — create a new change_type edit
      const newEdit: LabelEdit = {
        action: "change_type",
        row_id: action.row_id,
        label: action.label,
      };
      return { ...state, edits: [...state.edits, newEdit] };
    }

    case "clear_label": {
      // Collapse with existing add or change_type for the same row
      const existingIdx = state.edits.findIndex(
        (e) =>
          (e.action === "add" && e.id === action.row_id) ||
          (e.action === "change_type" && e.row_id === action.row_id),
      );

      if (existingIdx !== -1) {
        const existing = state.edits[existingIdx];
        if (existing.action === "add") {
          // Clear label on an add edit — remove the label field
          const updated: LabelEdit = { ...existing, label: undefined };
          const edits = [...state.edits];
          edits[existingIdx] = updated;
          return { ...state, edits };
        }
        // Collapse change_type → clear_label
        const updated: LabelEdit = {
          action: "clear_label",
          row_id: action.row_id,
        };
        const edits = [...state.edits];
        edits[existingIdx] = updated;
        return { ...state, edits };
      }

      // Also collapse with an existing clear_label (idempotent)
      const clearIdx = state.edits.findIndex(
        (e) => e.action === "clear_label" && e.row_id === action.row_id,
      );
      if (clearIdx !== -1) return state;

      const newEdit: LabelEdit = {
        action: "clear_label",
        row_id: action.row_id,
      };
      return { ...state, edits: [...state.edits, newEdit] };
    }

    case "select":
      return { ...state, selectedId: action.id };

    case "clear":
      return initialState;

    default:
      return state;
  }
}

/**
 * Returns true if two time ranges overlap (exclusive of touching boundaries).
 */
function overlaps(
  aStart: number,
  aEnd: number,
  bStart: number,
  bEnd: number,
): boolean {
  return aStart < bEnd && bStart < aEnd;
}

export function useLabelEdits(originalRows: DetectionRow[]) {
  const [state, dispatch] = useReducer(reducer, initialState);

  const mergedRows = useMemo(() => {
    // Build a mutable copy of originals
    const rows: DetectionRow[] = originalRows.map((r) => ({ ...r }));

    // Collect "add" edits — we'll need them to filter unlabeled rows
    const addEdits = state.edits.filter((e) => e.action === "add");

    // Apply moves
    for (const edit of state.edits) {
      if (edit.action === "move" && edit.row_id != null) {
        const row = rows.find((r) => r.row_id === edit.row_id);
        if (row && edit.start_utc != null && edit.end_utc != null) {
          row.start_utc = edit.start_utc;
          row.end_utc = edit.end_utc;
        }
      }
    }

    // Apply type changes — set label fields, clear others (single-label enforcement)
    for (const edit of state.edits) {
      if (edit.action === "change_type" && edit.row_id != null && edit.label) {
        const row = rows.find((r) => r.row_id === edit.row_id);
        if (row) {
          row.humpback = edit.label === "humpback" ? 1 : null;
          row.orca = edit.label === "orca" ? 1 : null;
          row.ship = edit.label === "ship" ? 1 : null;
          row.background = edit.label === "background" ? 1 : null;
        }
      }
      if (edit.action === "clear_label" && edit.row_id != null) {
        const row = rows.find((r) => r.row_id === edit.row_id);
        if (row) {
          row.humpback = null;
          row.orca = null;
          row.ship = null;
          row.background = null;
        }
      }
    }

    // Filter out deleted rows
    const deletedRowIds = new Set(
      state.edits
        .filter((e) => e.action === "delete" && e.row_id != null)
        .map((e) => e.row_id!),
    );
    const filteredRows = rows.filter((r) => !deletedRowIds.has(r.row_id));

    // Build new rows from "add" edits
    const newRows: DetectionRow[] = addEdits.map((edit) => ({
      row_id: edit.id ?? uuidv4(),
      start_utc: edit.start_utc ?? 0,
      end_utc: edit.end_utc ?? 0,
      avg_confidence: null,
      peak_confidence: null,
      n_windows: null,
      humpback: edit.label === "humpback" ? 1 : null,
      orca: edit.label === "orca" ? 1 : null,
      ship: edit.label === "ship" ? 1 : null,
      background: edit.label === "background" ? 1 : null,
    }));

    // Filter out unlabeled original rows that overlap with any newly added labeled row
    const labeledNewRows = newRows.filter(
      (r) =>
        r.humpback != null ||
        r.orca != null ||
        r.ship != null ||
        r.background != null,
    );

    function isUnlabeled(r: DetectionRow): boolean {
      return (
        r.humpback == null &&
        r.orca == null &&
        r.ship == null &&
        r.background == null
      );
    }

    const finalRows = filteredRows.filter((r) => {
      if (!isUnlabeled(r)) return true;
      // Remove unlabeled original rows overlapping with a new labeled row
      return !labeledNewRows.some((nr) =>
        overlaps(r.start_utc, r.end_utc, nr.start_utc, nr.end_utc),
      );
    });

    return [...finalRows, ...newRows];
  }, [originalRows, state.edits]);

  const isDirty = state.edits.length > 0;

  return {
    state,
    dispatch,
    mergedRows,
    isDirty,
    selectedId: state.selectedId,
  };
}

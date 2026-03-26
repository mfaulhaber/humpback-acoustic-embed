import { useReducer, useMemo } from "react";
import { v4 as uuidv4 } from "uuid";
import type { DetectionRow } from "@/api/types";
import type { LabelType } from "@/components/timeline/constants";

export type { LabelType };

export interface LabelEdit {
  action: "add" | "move" | "delete" | "change_type";
  id?: string;
  row_id?: string;
  start_sec?: number;
  end_sec?: number;
  new_start_sec?: number;
  new_end_sec?: number;
  label?: LabelType;
}

interface State {
  edits: LabelEdit[];
  selectedId: string | null;
}

export type Action =
  | { type: "add"; start_sec: number; end_sec: number; label: LabelType }
  | { type: "move"; row_id: string; new_start_sec: number; new_end_sec: number }
  | { type: "delete"; row_id: string }
  | { type: "change_type"; row_id: string; label: LabelType }
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
        start_sec: action.start_sec,
        end_sec: action.end_sec,
        label: action.label,
      };
      return {
        edits: [...state.edits, newEdit],
        selectedId: newId,
      };
    }

    case "move": {
      // Collapse with an existing move or add edit for the same row/id
      const existingIdx = state.edits.findIndex(
        (e) =>
          (e.action === "move" || e.action === "add") &&
          (e.row_id === action.row_id || e.id === action.row_id)
      );

      if (existingIdx !== -1) {
        const existing = state.edits[existingIdx];
        const updated: LabelEdit =
          existing.action === "add"
            ? {
                ...existing,
                start_sec: action.new_start_sec,
                end_sec: action.new_end_sec,
              }
            : {
                ...existing,
                new_start_sec: action.new_start_sec,
                new_end_sec: action.new_end_sec,
              };
        const edits = [...state.edits];
        edits[existingIdx] = updated;
        return { ...state, edits };
      }

      // No existing move/add — create a new move edit
      const newEdit: LabelEdit = {
        action: "move",
        row_id: action.row_id,
        new_start_sec: action.new_start_sec,
        new_end_sec: action.new_end_sec,
      };
      return { ...state, edits: [...state.edits, newEdit] };
    }

    case "delete": {
      // If deleting an "add" edit, just remove it from the list
      const addIdx = state.edits.findIndex(
        (e) => e.action === "add" && e.id === action.row_id
      );
      if (addIdx !== -1) {
        const edits = state.edits.filter((_, i) => i !== addIdx);
        return {
          edits,
          selectedId:
            state.selectedId === action.row_id ? null : state.selectedId,
        };
      }

      // Otherwise remove all prior edits for this row and add a delete edit
      const edits = state.edits.filter(
        (e) => e.row_id !== action.row_id && e.id !== action.row_id
      );
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

    case "change_type": {
      // Collapse with existing add or change_type for the same row/id
      const existingIdx = state.edits.findIndex(
        (e) =>
          (e.action === "add" || e.action === "change_type") &&
          (e.row_id === action.row_id || e.id === action.row_id)
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
  bEnd: number
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
        if (row && edit.new_start_sec != null && edit.new_end_sec != null) {
          row.start_sec = edit.new_start_sec;
          row.end_sec = edit.new_end_sec;
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
    }

    // Filter out deleted rows
    const deletedRowIds = new Set(
      state.edits
        .filter((e) => e.action === "delete" && e.row_id != null)
        .map((e) => e.row_id as string)
    );
    const filteredRows = rows.filter(
      (r) => r.row_id == null || !deletedRowIds.has(r.row_id)
    );

    // Build new rows from "add" edits
    const newRows: DetectionRow[] = addEdits.map((edit) => ({
      row_id: edit.id ?? null,
      filename: "",
      start_sec: edit.start_sec ?? 0,
      end_sec: edit.end_sec ?? 0,
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
        r.background != null
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
        overlaps(r.start_sec, r.end_sec, nr.start_sec, nr.end_sec)
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

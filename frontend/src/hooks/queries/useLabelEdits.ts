import { useReducer, useMemo } from "react";
import { v4 as uuidv4 } from "uuid";
import type { DetectionRow } from "@/api/types";
import type { LabelType } from "@/components/timeline/constants";

export type { LabelType };

export interface LabelEdit {
  action: "add" | "move" | "delete" | "change_type";
  id?: string;
  start_utc?: number;
  end_utc?: number;
  new_start_utc?: number;
  new_end_utc?: number;
  label?: LabelType;
}

interface State {
  edits: LabelEdit[];
  selectedId: string | null;
}

export type Action =
  | { type: "add"; start_utc: number; end_utc: number; label: LabelType }
  | { type: "move"; start_utc: number; end_utc: number; new_start_utc: number; new_end_utc: number }
  | { type: "delete"; start_utc: number; end_utc: number }
  | { type: "delete_by_id"; id: string }
  | { type: "change_type"; start_utc: number; end_utc: number; label: LabelType }
  | { type: "select"; id: string | null }
  | { type: "clear" };

function utcKey(startUtc: number, endUtc: number): string {
  return `${startUtc}:${endUtc}`;
}

const initialState: State = {
  edits: [],
  selectedId: null,
};

function editMatchesUtc(e: LabelEdit, startUtc: number, endUtc: number): boolean {
  return e.start_utc === startUtc && e.end_utc === endUtc;
}

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
      // Collapse with an existing move or add edit for the same UTC pair
      const existingIdx = state.edits.findIndex(
        (e) =>
          (e.action === "move" || e.action === "add") &&
          editMatchesUtc(e, action.start_utc, action.end_utc)
      );

      if (existingIdx !== -1) {
        const existing = state.edits[existingIdx];
        const updated: LabelEdit =
          existing.action === "add"
            ? {
                ...existing,
                start_utc: action.new_start_utc,
                end_utc: action.new_end_utc,
              }
            : {
                ...existing,
                new_start_utc: action.new_start_utc,
                new_end_utc: action.new_end_utc,
              };
        const edits = [...state.edits];
        edits[existingIdx] = updated;
        return { ...state, edits };
      }

      // No existing move/add — create a new move edit
      const newEdit: LabelEdit = {
        action: "move",
        start_utc: action.start_utc,
        end_utc: action.end_utc,
        new_start_utc: action.new_start_utc,
        new_end_utc: action.new_end_utc,
      };
      return { ...state, edits: [...state.edits, newEdit] };
    }

    case "delete": {
      const key = utcKey(action.start_utc, action.end_utc);
      // If deleting an "add" edit, just remove it from the list
      const addIdx = state.edits.findIndex(
        (e) => e.action === "add" && e.start_utc === action.start_utc && e.end_utc === action.end_utc
      );
      if (addIdx !== -1) {
        const editId = state.edits[addIdx].id;
        const edits = state.edits.filter((_, i) => i !== addIdx);
        return {
          edits,
          selectedId:
            state.selectedId === editId ? null : state.selectedId,
        };
      }

      // Otherwise remove all prior edits for this UTC pair and add a delete edit
      const edits = state.edits.filter(
        (e) => !editMatchesUtc(e, action.start_utc, action.end_utc)
      );
      const deleteEdit: LabelEdit = {
        action: "delete",
        start_utc: action.start_utc,
        end_utc: action.end_utc,
      };
      return {
        edits: [...edits, deleteEdit],
        selectedId:
          state.selectedId === key ? null : state.selectedId,
      };
    }

    case "delete_by_id": {
      // Delete an "add" edit by its generated id
      const addIdx = state.edits.findIndex(
        (e) => e.action === "add" && e.id === action.id
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
      // Collapse with existing add or change_type for the same UTC pair
      const existingIdx = state.edits.findIndex(
        (e) =>
          (e.action === "add" || e.action === "change_type") &&
          editMatchesUtc(e, action.start_utc, action.end_utc)
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
        start_utc: action.start_utc,
        end_utc: action.end_utc,
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
      if (edit.action === "move" && edit.start_utc != null && edit.end_utc != null) {
        const row = rows.find((r) => r.start_utc === edit.start_utc && r.end_utc === edit.end_utc);
        if (row && edit.new_start_utc != null && edit.new_end_utc != null) {
          row.start_utc = edit.new_start_utc;
          row.end_utc = edit.new_end_utc;
        }
      }
    }

    // Apply type changes — set label fields, clear others (single-label enforcement)
    for (const edit of state.edits) {
      if (edit.action === "change_type" && edit.start_utc != null && edit.end_utc != null && edit.label) {
        const row = rows.find((r) => r.start_utc === edit.start_utc && r.end_utc === edit.end_utc);
        if (row) {
          row.humpback = edit.label === "humpback" ? 1 : null;
          row.orca = edit.label === "orca" ? 1 : null;
          row.ship = edit.label === "ship" ? 1 : null;
          row.background = edit.label === "background" ? 1 : null;
        }
      }
    }

    // Filter out deleted rows
    const deletedKeys = new Set(
      state.edits
        .filter((e) => e.action === "delete" && e.start_utc != null && e.end_utc != null)
        .map((e) => utcKey(e.start_utc!, e.end_utc!))
    );
    const filteredRows = rows.filter(
      (r) => !deletedKeys.has(utcKey(r.start_utc, r.end_utc))
    );

    // Build new rows from "add" edits
    const newRows: DetectionRow[] = addEdits.map((edit) => ({
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
        overlaps(r.start_utc, r.end_utc, nr.start_utc, nr.end_utc)
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

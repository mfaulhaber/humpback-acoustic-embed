import type { LabelingAnnotation } from "@/api/types";
import { Trash2 } from "lucide-react";

interface Props {
  annotations: LabelingAnnotation[];
  highlightedId: string | null;
  onSelect: (id: string | null) => void;
  onDelete: (id: string) => void;
}

export function AnnotationList({
  annotations,
  highlightedId,
  onSelect,
  onDelete,
}: Props) {
  if (annotations.length === 0) {
    return (
      <div className="text-xs text-slate-400 italic py-1">
        No annotations. Click and drag on the spectrogram to create one.
      </div>
    );
  }

  return (
    <table className="w-full text-xs">
      <thead>
        <tr className="text-slate-500 border-b">
          <th className="text-left py-1 pr-2">Range</th>
          <th className="text-left py-1 pr-2">Label</th>
          <th className="text-left py-1 pr-2">Notes</th>
          <th className="w-6" />
        </tr>
      </thead>
      <tbody>
        {annotations.map((ann) => (
          <tr
            key={ann.id}
            onClick={() => onSelect(ann.id === highlightedId ? null : ann.id)}
            className={`cursor-pointer border-b border-slate-50 ${
              ann.id === highlightedId
                ? "bg-blue-50"
                : "hover:bg-slate-50"
            }`}
          >
            <td className="py-1 pr-2 font-mono text-slate-600">
              {ann.start_offset_sec.toFixed(2)}s &ndash;{" "}
              {ann.end_offset_sec.toFixed(2)}s
            </td>
            <td className="py-1 pr-2">
              <span className="px-1.5 py-0.5 bg-violet-100 text-violet-700 rounded text-[10px] font-medium">
                {ann.label}
              </span>
            </td>
            <td className="py-1 pr-2 text-slate-500 truncate max-w-[120px]">
              {ann.notes ?? ""}
            </td>
            <td className="py-1">
              <button
                onClick={(e) => {
                  e.stopPropagation();
                  onDelete(ann.id);
                }}
                className="p-0.5 rounded hover:bg-red-100 text-slate-400 hover:text-red-500"
                title="Delete annotation"
              >
                <Trash2 className="h-3 w-3" />
              </button>
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

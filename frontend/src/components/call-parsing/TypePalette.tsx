import { useState } from "react";
import { Plus } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { useVocalizationTypes } from "@/hooks/queries/useVocalization";
import { createVocalizationType } from "@/api/client";

// Simple deterministic color from type name
function typeColor(name: string): string {
  let hash = 0;
  for (let i = 0; i < name.length; i++) {
    hash = name.charCodeAt(i) + ((hash << 5) - hash);
  }
  const hue = Math.abs(hash) % 360;
  return `hsl(${hue}, 60%, 45%)`;
}

export { typeColor };

interface TypePaletteProps {
  activeType: string | null; // null = no selection, "" = negative
  onSelectType: (typeName: string | null) => void;
}

export function TypePalette({ activeType, onSelectType }: TypePaletteProps) {
  const { data: vocTypes = [], refetch } = useVocalizationTypes();
  const [showAddInput, setShowAddInput] = useState(false);
  const [newTypeName, setNewTypeName] = useState("");

  const handleAddType = async () => {
    const trimmed = newTypeName.trim();
    if (!trimmed) return;
    try {
      await createVocalizationType({ name: trimmed, description: null });
      await refetch();
      setNewTypeName("");
      setShowAddInput(false);
      onSelectType(trimmed);
    } catch {
      /* ignore */
    }
  };

  return (
    <div className="flex flex-wrap items-center gap-1.5 px-4 py-2 border-t">
      {/* None status indicator — non-interactive; reserves layout space
          even when hidden so the palette does not reflow between events. */}
      <span
        data-testid="palette-none-indicator"
        aria-hidden={activeType !== null}
        className="px-2.5 py-1 rounded-full text-xs font-medium border border-dashed border-muted-foreground/40 text-muted-foreground select-none cursor-default"
        style={{ visibility: activeType === null ? "visible" : "hidden" }}
      >
        None
      </span>

      {/* Negative option */}
      <button
        className={`px-2.5 py-1 rounded-full text-xs font-medium border transition-all ${
          activeType === ""
            ? "border-red-500 bg-red-50 text-red-600 ring-2 ring-red-300"
            : "border-red-300 text-red-500 hover:border-red-400"
        }`}
        onClick={() => onSelectType("")}
      >
        (Negative)
      </button>

      {/* Type buttons */}
      {vocTypes.map((vt) => (
        <button
          key={vt.id}
          className={`px-2.5 py-1 rounded-full text-xs font-medium border transition-all ${
            activeType === vt.name
              ? "ring-2 ring-offset-1"
              : "hover:opacity-80"
          }`}
          style={{
            borderColor: typeColor(vt.name),
            color: activeType === vt.name ? "white" : typeColor(vt.name),
            backgroundColor:
              activeType === vt.name ? typeColor(vt.name) : "transparent",
            ["--tw-ring-color" as string]: typeColor(vt.name),
          }}
          onClick={() => onSelectType(vt.name)}
        >
          {vt.name}
        </button>
      ))}

      {/* Add type */}
      {showAddInput ? (
        <div className="flex items-center gap-1">
          <Input
            className="h-7 w-32 text-xs"
            placeholder="New type name"
            value={newTypeName}
            onChange={(e) => setNewTypeName(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter") void handleAddType();
              if (e.key === "Escape") {
                setShowAddInput(false);
                setNewTypeName("");
              }
            }}
            autoFocus
          />
          <Button
            size="sm"
            variant="ghost"
            className="h-7 text-xs"
            onClick={() => void handleAddType()}
          >
            Add
          </Button>
        </div>
      ) : (
        <button
          className="px-2.5 py-1 rounded-full text-xs font-medium border border-dashed border-slate-300 text-slate-400 hover:border-slate-500 hover:text-slate-600 flex items-center gap-1"
          onClick={() => setShowAddInput(true)}
        >
          <Plus className="h-3 w-3" /> Add Type
        </button>
      )}
    </div>
  );
}

import { useEffect, useMemo } from "react";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";

interface ModelFilterProps {
  items: Array<{ model_version: string }>;
  value: string;
  onChange: (v: string) => void;
}

export function ModelFilter({ items, value, onChange }: ModelFilterProps) {
  const models = useMemo(() => {
    const unique = [...new Set(items.map((i) => i.model_version))].sort();
    return unique;
  }, [items]);

  // Auto-select first model when current value is invalid
  useEffect(() => {
    if (models.length > 0 && !models.includes(value)) {
      onChange(models[0]);
    }
  }, [models, value, onChange]);

  if (models.length === 0) return null;

  return (
    <div className="flex items-center gap-2">
      <label className="text-sm font-medium whitespace-nowrap">Model:</label>
      <Select value={value} onValueChange={onChange}>
        <SelectTrigger className="w-48 h-8 text-xs">
          <SelectValue placeholder="Select model..." />
        </SelectTrigger>
        <SelectContent>
          {models.map((m) => (
            <SelectItem key={m} value={m}>
              {m}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
    </div>
  );
}

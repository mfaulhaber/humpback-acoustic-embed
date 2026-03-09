import { useState, useEffect } from "react";
import { Calendar } from "lucide-react";
import { type DateRange } from "react-day-picker";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover";
import { Calendar as CalendarComponent } from "@/components/ui/calendar";
import { Separator } from "@/components/ui/separator";
import { cn } from "@/lib/utils";

export interface DateRangeUtc {
  startEpoch: number | null; // UTC epoch seconds
  endEpoch: number | null;
}

interface DateRangePickerUtcProps {
  value: DateRangeUtc;
  onChange: (range: DateRangeUtc) => void;
  placeholder?: string;
  className?: string;
  disabled?: boolean;
}

/** Build a "fake-local" Date where local fields match UTC intent. */
function fakeLocalFromUtcEpoch(epoch: number): Date {
  const d = new Date(epoch * 1000);
  return new Date(d.getUTCFullYear(), d.getUTCMonth(), d.getUTCDate());
}

function timeStringFromUtcEpoch(epoch: number): string {
  const d = new Date(epoch * 1000);
  const hh = String(d.getUTCHours()).padStart(2, "0");
  const mm = String(d.getUTCMinutes()).padStart(2, "0");
  return `${hh}:${mm}`;
}

function formatDateUtc(epoch: number): string {
  const d = new Date(epoch * 1000);
  const p = (n: number) => String(n).padStart(2, "0");
  return `${d.getUTCFullYear()}-${p(d.getUTCMonth() + 1)}-${p(d.getUTCDate())}`;
}

function formatDisplayRange(value: DateRangeUtc): string {
  if (value.startEpoch == null || value.endEpoch == null) return "";
  const fmtDate = (e: number) => formatDateUtc(e);
  const fmtTime = (e: number) => timeStringFromUtcEpoch(e);
  return `${fmtDate(value.startEpoch)} ${fmtTime(value.startEpoch)} — ${fmtDate(value.endEpoch)} ${fmtTime(value.endEpoch)} UTC`;
}

function combineEpoch(fakeLocalDate: Date, timeStr: string): number | null {
  const match = timeStr.match(/^(\d{2}):(\d{2})$/);
  if (!match) return null;
  const hours = Number(match[1]);
  const minutes = Number(match[2]);
  if (hours > 23 || minutes > 59) return null;
  const ms = Date.UTC(
    fakeLocalDate.getFullYear(),
    fakeLocalDate.getMonth(),
    fakeLocalDate.getDate(),
    hours,
    minutes,
  );
  return ms / 1000;
}

export function DateRangePickerUtc({
  value,
  onChange,
  placeholder = "Select date range (UTC)",
  className,
  disabled = false,
}: DateRangePickerUtcProps) {
  const [open, setOpen] = useState(false);
  const [draftRange, setDraftRange] = useState<DateRange | undefined>(undefined);
  const [draftStartTime, setDraftStartTime] = useState("00:00");
  const [draftEndTime, setDraftEndTime] = useState("00:00");
  const [month, setMonth] = useState<Date>(new Date());

  // Sync draft from value when popover opens
  useEffect(() => {
    if (open) {
      if (value.startEpoch != null && value.endEpoch != null) {
        setDraftRange({
          from: fakeLocalFromUtcEpoch(value.startEpoch),
          to: fakeLocalFromUtcEpoch(value.endEpoch),
        });
        setDraftStartTime(timeStringFromUtcEpoch(value.startEpoch));
        setDraftEndTime(timeStringFromUtcEpoch(value.endEpoch));
        setMonth(fakeLocalFromUtcEpoch(value.startEpoch));
      } else {
        setDraftRange(undefined);
        setDraftStartTime("00:00");
        setDraftEndTime("00:00");
      }
    }
  }, [open, value.startEpoch, value.endEpoch]);

  const handleApply = () => {
    if (!draftRange?.from || !draftRange?.to) return;
    const startEpoch = combineEpoch(draftRange.from, draftStartTime);
    const endEpoch = combineEpoch(draftRange.to, draftEndTime);
    if (startEpoch == null || endEpoch == null) return;
    onChange({ startEpoch, endEpoch });
    setOpen(false);
  };

  const handleCancel = () => {
    setOpen(false);
  };

  const canApply =
    draftRange?.from != null &&
    draftRange?.to != null &&
    /^\d{2}:\d{2}$/.test(draftStartTime) &&
    /^\d{2}:\d{2}$/.test(draftEndTime);

  const displayText = formatDisplayRange(value);

  const formatFakeLocalDate = (d: Date) => {
    const p = (n: number) => String(n).padStart(2, "0");
    return `${d.getFullYear()}-${p(d.getMonth() + 1)}-${p(d.getDate())}`;
  };

  return (
    <Popover open={open} onOpenChange={setOpen}>
      <PopoverTrigger asChild>
        <Button
          variant="outline"
          className={cn(
            "w-full justify-start text-left font-mono",
            !displayText && "text-muted-foreground",
            className,
          )}
          disabled={disabled}
          data-testid="date-range-trigger"
        >
          <Calendar className="mr-2 h-4 w-4" />
          {displayText || placeholder}
        </Button>
      </PopoverTrigger>
      <PopoverContent className="w-auto p-0" align="start">
        <div className="p-3">
          <CalendarComponent
            mode="range"
            numberOfMonths={2}
            selected={draftRange}
            onSelect={setDraftRange}
            month={month}
            onMonthChange={setMonth}
          />
        </div>
        <Separator />
        <div className="grid grid-cols-2 gap-4 p-4">
          <div>
            <label className="text-xs font-medium text-muted-foreground">Start</label>
            <p className="text-sm font-mono mt-0.5">
              {draftRange?.from ? formatFakeLocalDate(draftRange.from) : "—"}
            </p>
            <Input
              type="text"
              placeholder="HH:MM"
              value={draftStartTime}
              onChange={(e) => setDraftStartTime(e.target.value)}
              className="mt-1 w-24 font-mono text-sm"
              data-testid="start-time-input"
            />
          </div>
          <div>
            <label className="text-xs font-medium text-muted-foreground">End</label>
            <p className="text-sm font-mono mt-0.5">
              {draftRange?.to ? formatFakeLocalDate(draftRange.to) : "—"}
            </p>
            <Input
              type="text"
              placeholder="HH:MM"
              value={draftEndTime}
              onChange={(e) => setDraftEndTime(e.target.value)}
              className="mt-1 w-24 font-mono text-sm"
              data-testid="end-time-input"
            />
          </div>
        </div>
        <p className="px-4 pb-2 text-xs text-muted-foreground">All times are UTC.</p>
        <Separator />
        <div className="flex justify-end gap-2 p-3">
          <Button variant="ghost" size="sm" onClick={handleCancel}>
            Cancel
          </Button>
          <Button size="sm" onClick={handleApply} disabled={!canApply} data-testid="date-range-apply">
            Apply
          </Button>
        </div>
      </PopoverContent>
    </Popover>
  );
}

export function shortId(id: string): string {
  return id.slice(0, 8);
}

export function fmtDate(iso: string): string {
  return new Date(iso).toLocaleString();
}

export function formatTime(seconds: number): string {
  const h = Math.floor(seconds / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  const s = Math.floor(seconds % 60);
  if (h > 0)
    return `${h}:${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`;
  return `${m}:${String(s).padStart(2, "0")}`;
}

/** Format seconds as `m:s.s` with one decimal place. */
export function formatTimeDecimal(seconds: number): string {
  const m = Math.floor(seconds / 60);
  const s = seconds % 60;
  return `${m}:${s.toFixed(1).padStart(4, "0")}`;
}

export function jsonPretty(obj: unknown): string {
  return JSON.stringify(obj, null, 2);
}

export function audioDisplayName(filename: string, folderPath?: string): string {
  if (folderPath) return `${folderPath}/${filename}`;
  return filename;
}

/** Format an offset (seconds from job start) as recording-based UTC time HH:MM:SS.d */
export function formatRecordingTime(
  offsetSec: number,
  jobStartEpoch: number,
): string {
  const epoch = jobStartEpoch + offsetSec;
  const d = new Date(epoch * 1000);
  const hh = String(d.getUTCHours()).padStart(2, "0");
  const mm = String(d.getUTCMinutes()).padStart(2, "0");
  const ss = d.getUTCSeconds() + (epoch % 1);
  return `${hh}:${mm}:${ss.toFixed(1).padStart(4, "0")}`;
}

/** Format a UTC epoch (seconds) as a short month+day string, e.g. "Apr 14". */
export function formatUtcShort(epoch: number): string {
  const d = new Date(epoch * 1000);
  const months = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
  ];
  return `${months[d.getUTCMonth()]} ${d.getUTCDate()}`;
}

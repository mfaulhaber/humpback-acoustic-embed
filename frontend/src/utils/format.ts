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

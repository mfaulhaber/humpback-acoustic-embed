export function shortId(id: string): string {
  return id.slice(0, 8);
}

export function fmtDate(iso: string): string {
  return new Date(iso).toLocaleString();
}

export function formatTime(seconds: number): string {
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${m}:${s.toString().padStart(2, "0")}`;
}

export function jsonPretty(obj: unknown): string {
  return JSON.stringify(obj, null, 2);
}

export function audioDisplayName(filename: string, folderPath?: string): string {
  if (folderPath) return `${folderPath}/${filename}`;
  return filename;
}

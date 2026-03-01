export const AUDIO_EXTENSIONS = [".wav", ".mp3", ".flac"];

export function isAudioFile(filename: string): boolean {
  const lower = filename.toLowerCase();
  return AUDIO_EXTENSIONS.some((ext) => lower.endsWith(ext));
}

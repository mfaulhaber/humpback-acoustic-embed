/**
 * Deterministic per-motif color assignment used when many motifs are
 * highlighted on the timeline at once (Token Count: 2 | 3 | 4 selector on
 * the masked-transformer detail page).
 *
 * Same ``motif_key`` always yields the same palette entry across renders
 * and across page reloads. Distinct keys map to distinct entries when the
 * palette is large enough.
 */

export interface MotifColor {
  /** Background fill (uniform alpha across all motifs). */
  fill: string;
  /** Border stroke at higher alpha. */
  border: string;
}

/**
 * 12-hue palette spanning the color wheel at fixed saturation/lightness.
 * Uses ``hsla`` so alpha is explicit; fill alpha is constant so all
 * motifs render at the same visual weight.
 */
export const MOTIF_PALETTE: readonly MotifColor[] = [
  { fill: "hsla(  0, 70%, 55%, 0.30)", border: "hsla(  0, 70%, 45%, 0.85)" },
  { fill: "hsla( 30, 80%, 50%, 0.30)", border: "hsla( 30, 80%, 40%, 0.85)" },
  { fill: "hsla( 55, 80%, 45%, 0.30)", border: "hsla( 55, 80%, 35%, 0.85)" },
  { fill: "hsla( 90, 60%, 45%, 0.30)", border: "hsla( 90, 60%, 35%, 0.85)" },
  { fill: "hsla(140, 55%, 45%, 0.30)", border: "hsla(140, 55%, 35%, 0.85)" },
  { fill: "hsla(170, 60%, 45%, 0.30)", border: "hsla(170, 60%, 35%, 0.85)" },
  { fill: "hsla(200, 70%, 50%, 0.30)", border: "hsla(200, 70%, 40%, 0.85)" },
  { fill: "hsla(225, 70%, 60%, 0.30)", border: "hsla(225, 70%, 50%, 0.85)" },
  { fill: "hsla(260, 60%, 60%, 0.30)", border: "hsla(260, 60%, 50%, 0.85)" },
  { fill: "hsla(290, 60%, 55%, 0.30)", border: "hsla(290, 60%, 45%, 0.85)" },
  { fill: "hsla(320, 65%, 55%, 0.30)", border: "hsla(320, 65%, 45%, 0.85)" },
  { fill: "hsla(345, 70%, 60%, 0.30)", border: "hsla(345, 70%, 50%, 0.85)" },
] as const;

/**
 * djb2 string hash — stable across JS engines, returns an unsigned int.
 */
function hashString(s: string): number {
  let h = 5381;
  for (let i = 0; i < s.length; i++) {
    h = (h * 33) ^ s.charCodeAt(i);
  }
  return h >>> 0;
}

export function colorForMotifKey(motifKey: string): MotifColor {
  const idx = hashString(motifKey) % MOTIF_PALETTE.length;
  return MOTIF_PALETTE[idx];
}

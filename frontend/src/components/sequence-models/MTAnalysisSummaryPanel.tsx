import type { MaskedTransformerAnalysisReport } from "@/api/sequenceModels";

type SummaryTone = "good" | "warn" | "bad" | "neutral";

interface SummaryLine {
  tone: SummaryTone;
  text: string;
}

interface SpaceDiagnostic {
  tone: SummaryTone;
  title: string;
  lines: string[];
}

export interface MTAnalysisSummary {
  run: SummaryLine[];
  geometry: SummaryLine[];
  spaceDiagnostics: SpaceDiagnostic[];
  neighborhood: SummaryLine[];
}

const SPACE_DIAGNOSTIC_ORDER = [
  "contextual.raw_l2",
  "retrieval.raw_l2",
  "contextual.centered_l2",
  "retrieval.centered_l2",
  "contextual.remove_pc1",
  "retrieval.remove_pc1",
  "contextual.remove_pc3",
  "retrieval.remove_pc3",
  "contextual.remove_pc5",
  "retrieval.remove_pc5",
  "contextual.whiten_pca",
  "retrieval.whiten_pca",
  "contextual.remove_pc10",
  "retrieval.remove_pc10",
];

function toneClass(tone: SummaryTone): string {
  if (tone === "good") {
    return "border-emerald-200 bg-emerald-50 text-emerald-950";
  }
  if (tone === "warn") return "border-amber-200 bg-amber-50 text-amber-950";
  if (tone === "bad") return "border-red-200 bg-red-50 text-red-950";
  return "border-slate-200 bg-slate-50 text-slate-900";
}

function formatMetric(value: unknown): string {
  if (typeof value !== "number" || !Number.isFinite(value)) return "-";
  if (Math.abs(value) >= 100) return value.toLocaleString();
  return value.toFixed(3);
}

function formatRank(value: unknown): string {
  if (typeof value !== "number" || !Number.isFinite(value)) return "-";
  return value.toLocaleString(undefined, {
    maximumFractionDigits: 1,
    minimumFractionDigits: 1,
  });
}

function formatPercent(value: unknown): string {
  if (typeof value !== "number" || !Number.isFinite(value)) return "-";
  return `${Math.round(value * 100)}%`;
}

function formatCount(value: unknown): string {
  if (typeof value !== "number" || !Number.isFinite(value)) return "-";
  return Math.round(value).toLocaleString();
}

function formatRange(values: number[]): string {
  if (values.length === 0) return "-";
  const min = Math.min(...values);
  const max = Math.max(...values);
  return `${formatMetric(min)}-${formatMetric(max)}`;
}

function asNumber(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function warningsFor(row: Record<string, unknown>): string[] {
  return Array.isArray(row.warnings)
    ? row.warnings.filter((item): item is string => typeof item === "string")
    : [];
}

function isGoodMeanBand(value: unknown): boolean {
  return value === "good" || value === "okay";
}

function isGoodRankBand(value: unknown): boolean {
  return value === "broad" || value === "plausible";
}

function isPoorMeanBand(value: unknown): boolean {
  return value === "suspicious" || value === "collapse_risk";
}

function isPoorRankBand(value: unknown): boolean {
  return value === "weak" || value === "severe_collapse";
}

function isOkayRankBand(value: unknown): boolean {
  return value === "weak";
}

function titleizeToken(value: string): string {
  return value
    .split("_")
    .filter(Boolean)
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(" ");
}

function variantLabel(variant: string): string {
  if (variant === "raw_l2") return "Raw";
  if (variant === "centered_l2") return "Centered";
  if (variant === "whiten_pca") return "Whitened";
  const match = /^remove_pc(\d+)$/.exec(variant);
  if (match) return `PC${match[1]}-Removed`;
  return titleizeToken(variant);
}

function spaceTitle(space: string): string {
  const [sourceSpace, variant = ""] = space.split(".");
  return `${variantLabel(variant)} ${titleizeToken(sourceSpace)} Space`;
}

function nestedRecord(
  row: Record<string, unknown>,
  key: string,
): Record<string, unknown> {
  const value = row[key];
  return value && typeof value === "object" && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : {};
}

function saturationLine(row: Record<string, unknown>): string {
  const percentiles = nestedRecord(row, "random_pair_percentiles");
  const p75 = asNumber(percentiles.p75);
  const p95 = asNumber(percentiles.p95);
  if (p75 == null && p95 == null) {
    return "Random-pair cosine percentiles were not reported.";
  }
  const label =
    (p75 != null && p75 > 0.7) || (p95 != null && p95 > 0.95)
      ? "Very saturated"
      : (p75 != null && p75 > 0.3) || (p95 != null && p95 > 0.7)
        ? "Some saturation"
        : "Low saturation";
  return `${label}: random-pair cosine p75 ${formatMetric(
    p75,
  )}, p95 ${formatMetric(p95)}.`;
}

function assessmentLine(row: Record<string, unknown>): string {
  if (row.available === false) {
    const reason = row.reason ? String(row.reason) : "not available";
    return `This space is unavailable: ${reason}.`;
  }
  if (
    warningsFor(row).length > 0 ||
    isPoorMeanBand(row.mean_vector_band) ||
    row.effective_rank_band === "severe_collapse"
  ) {
    return "Not a good space to trust directly.";
  }
  if (
    isGoodMeanBand(row.mean_vector_band) &&
    isGoodRankBand(row.effective_rank_band)
  ) {
    return "Good candidate for inference and motif search.";
  }
  if (isOkayRankBand(row.effective_rank_band)) {
    return "Usable with caution; prefer a broader space when available.";
  }
  return "Interpret with caution until this measurement has a clearer threshold.";
}

function diagnosticTone(row: Record<string, unknown>): SummaryTone {
  if (row.available === false) return "warn";
  if (
    warningsFor(row).length > 0 ||
    isPoorMeanBand(row.mean_vector_band) ||
    row.effective_rank_band === "severe_collapse"
  ) {
    return "bad";
  }
  if (isPoorRankBand(row.effective_rank_band)) return "warn";
  if (
    isGoodMeanBand(row.mean_vector_band) &&
    isGoodRankBand(row.effective_rank_band)
  ) {
    return "good";
  }
  return "neutral";
}

function buildSpaceDiagnostic(
  space: string,
  row: Record<string, unknown>,
): SpaceDiagnostic {
  if (row.available === false) {
    return {
      tone: "warn",
      title: spaceTitle(space),
      lines: [assessmentLine(row)],
    };
  }

  const pca = nestedRecord(row, "pca_explained_variance");
  const rankBand = row.effective_rank_band
    ? String(row.effective_rank_band)
    : "unknown";
  const vectorDim = row.vector_dim;
  const lines = [
    saturationLine(row),
    `Mean vector band: ${String(row.mean_vector_band ?? "unknown")}.`,
    `Effective rank: ${formatRank(
      row.effective_rank,
    )}, ${rankBand} for ${formatCount(vectorDim)} dims.`,
    `PC1 alone explains ${formatPercent(pca.pc1)} of variance.`,
    assessmentLine(row),
  ];

  return {
    tone: diagnosticTone(row),
    title: spaceTitle(space),
    lines,
  };
}

function orderedSpaceEntries(
  spaces: Record<string, Record<string, unknown>>,
): [string, Record<string, unknown>][] {
  const ordered: [string, Record<string, unknown>][] =
    SPACE_DIAGNOSTIC_ORDER.flatMap((space) => {
      const row = spaces[space];
      return row ? [[space, row]] : [];
    });
  const seen = new Set(ordered.map(([space]) => space));
  const remaining = Object.entries(spaces)
    .filter(([space]) => !seen.has(space))
    .sort(([a], [b]) => a.localeCompare(b));
  return [...ordered, ...remaining];
}

function metricValues(
  report: MaskedTransformerAnalysisReport,
  metric: string,
): number[] {
  return Object.values(report.results).flatMap((variants) =>
    Object.values(variants).flatMap((metrics) => {
      const value = metrics[metric];
      return typeof value === "number" && Number.isFinite(value) ? [value] : [];
    }),
  );
}

export function buildMTAnalysisSummary(
  report: MaskedTransformerAnalysisReport,
): MTAnalysisSummary {
  const run: SummaryLine[] = [
    {
      tone: "neutral",
      text: `k=${formatMetric(report.job.k)} across ${formatMetric(
        report.job.total_sequences,
      )} sequences and ${formatMetric(report.job.total_chunks)} chunks.`,
    },
    {
      tone: "neutral",
      text: `Final loss: train ${formatMetric(
        report.job.final_train_loss,
      )}, validation ${formatMetric(report.job.final_val_loss)}.`,
    },
  ];

  const geometrySummary = report.geometry_report?.summary ?? {};
  const spaces = report.geometry_report?.spaces ?? {};
  const saturated = geometrySummary.retrieval_raw_saturated === true;
  const lambdaBlocked = geometrySummary.lambda_sweeps_blocked === true;
  const warningCount = Array.isArray(geometrySummary.warnings)
    ? geometrySummary.warnings.length
    : 0;
  const geometry: SummaryLine[] = [];
  if (!report.geometry_report) {
    geometry.push({
      tone: "warn",
      text: "Geometry diagnostics were not included in this report.",
    });
  } else if (saturated || lambdaBlocked || warningCount > 0) {
    const flags = [
      saturated ? "raw retrieval saturation" : null,
      lambdaBlocked ? "lambda sweeps blocked" : null,
      warningCount > 0 ? `${warningCount} geometry warnings` : null,
    ].filter(Boolean);
    geometry.push({
      tone: saturated || lambdaBlocked ? "bad" : "warn",
      text: `Raw spaces need caution: ${flags.join(", ")}.`,
    });
  } else {
    geometry.push({
      tone: "good",
      text: "No geometry summary warnings were reported.",
    });
  }

  const spaceDiagnostics = report.geometry_report
    ? orderedSpaceEntries(spaces).map(([space, row]) =>
        buildSpaceDiagnostic(space, row),
      )
    : [];

  const sameToken = metricValues(report, "same_token");
  const similarDuration = metricValues(report, "similar_duration");
  const sameRegion = metricValues(report, "same_region");
  const adjacent = metricValues(report, "adjacent_1s");
  const nearby = metricValues(report, "nearby_5s");
  const neighborhood: SummaryLine[] = [
    {
      tone: "neutral",
      text: `Same-token neighbor rate range: ${formatRange(sameToken)}.`,
    },
    {
      tone: "neutral",
      text: `Similar-duration neighbor rate range: ${formatRange(similarDuration)}.`,
    },
    {
      tone: "neutral",
      text: `Region/time proximity ranges: same-region ${formatRange(
        sameRegion,
      )}, adjacent-1s ${formatRange(adjacent)}, nearby-5s ${formatRange(nearby)}.`,
    },
  ];

  return {
    run,
    geometry,
    spaceDiagnostics,
    neighborhood,
  };
}

function SummarySection({
  title,
  lines,
}: {
  title: string;
  lines: SummaryLine[];
}) {
  return (
    <section className="space-y-2">
      <h3 className="text-xs font-semibold uppercase text-muted-foreground">
        {title}
      </h3>
      <div className="space-y-1.5">
        {lines.map((line, index) => (
          <div
            key={`${title}-${index}`}
            className={`rounded-md border px-3 py-2 text-xs ${toneClass(line.tone)}`}
          >
            {line.text}
          </div>
        ))}
      </div>
    </section>
  );
}

function SpaceDiagnosticsSection({
  spaces,
}: {
  spaces: SpaceDiagnostic[];
}) {
  if (spaces.length === 0) return null;
  return (
    <section className="space-y-2 lg:col-span-2">
      <h3 className="text-xs font-semibold uppercase text-muted-foreground">
        Embedding Spaces
      </h3>
      <div className="grid gap-2 lg:grid-cols-2">
        {spaces.map((space) => (
          <article
            key={space.title}
            className={`rounded-md border px-3 py-2 text-xs ${toneClass(
              space.tone,
            )}`}
          >
            <h4 className="mb-2 text-sm font-semibold">{space.title}</h4>
            <div className="space-y-1">
              {space.lines.map((line) => (
                <p key={line}>{line}</p>
              ))}
            </div>
          </article>
        ))}
      </div>
    </section>
  );
}

export function MTAnalysisSummaryPanel({
  report,
}: {
  report: MaskedTransformerAnalysisReport;
}) {
  const summary = buildMTAnalysisSummary(report);
  return (
    <div
      className="rounded-md border p-4"
      data-testid="mt-analysis-summary-panel"
    >
      <h2 className="mb-3 text-sm font-semibold">Summary</h2>
      <div className="grid gap-4 lg:grid-cols-2">
        <SummarySection title="Run" lines={summary.run} />
        <SummarySection title="Geometry" lines={summary.geometry} />
        <SpaceDiagnosticsSection spaces={summary.spaceDiagnostics} />
        <div className="lg:col-span-2">
          <SummarySection
            title="Neighborhood Behavior"
            lines={summary.neighborhood}
          />
        </div>
      </div>
    </div>
  );
}

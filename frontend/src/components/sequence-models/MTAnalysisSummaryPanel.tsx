import type { MaskedTransformerAnalysisReport } from "@/api/sequenceModels";

type SummaryTone = "good" | "warn" | "bad" | "neutral";

interface SummaryLine {
  tone: SummaryTone;
  text: string;
}

export interface MTAnalysisSummary {
  run: SummaryLine[];
  geometry: SummaryLine[];
  recommendedSpaces: SummaryLine[];
  cautionSpaces: SummaryLine[];
  neighborhood: SummaryLine[];
}

const RECOMMENDED_SPACE_ORDER = [
  "contextual.whiten_pca",
  "contextual.remove_pc10",
  "retrieval.whiten_pca",
  "retrieval.remove_pc10",
  "contextual.remove_pc5",
  "retrieval.remove_pc5",
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

function spaceSummary(space: string, row: Record<string, unknown>): string {
  const warningCount = warningsFor(row).length;
  const rank = asNumber(row.effective_rank);
  const parts = [
    String(row.mean_vector_band ?? "unknown mean"),
    String(row.effective_rank_band ?? "unknown rank"),
  ];
  if (rank != null) parts.push(`rank ${formatMetric(rank)}`);
  parts.push(warningCount === 0 ? "no warnings" : `${warningCount} warnings`);
  return `${space}: ${parts.join(", ")}`;
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
  const spaceEntries = Object.entries(spaces);
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

  const recommendedSpaces: SummaryLine[] = RECOMMENDED_SPACE_ORDER.flatMap(
    (space) => {
      const row = spaces[space];
      if (!row) return [];
      if (
        row.available === true &&
        warningsFor(row).length === 0 &&
        isGoodMeanBand(row.mean_vector_band) &&
        isGoodRankBand(row.effective_rank_band)
      ) {
        return [{ tone: "good" as const, text: spaceSummary(space, row) }];
      }
      return [];
    },
  );
  if (recommendedSpaces.length === 0) {
    recommendedSpaces.push({
      tone: "warn",
      text: "No requested embedding space is clean enough to recommend automatically.",
    });
  }

  const cautionSpaces: SummaryLine[] = spaceEntries.flatMap(([space, row]) => {
    if (
      row.available === false ||
      warningsFor(row).length > 0 ||
      isPoorMeanBand(row.mean_vector_band) ||
      isPoorRankBand(row.effective_rank_band)
    ) {
      return [
        {
          tone: row.available === false ? ("warn" as const) : ("bad" as const),
          text: spaceSummary(space, row),
        },
      ];
    }
    return [];
  });
  if (cautionSpaces.length === 0) {
    cautionSpaces.push({
      tone: "good",
      text: "No requested embedding spaces were flagged for caution.",
    });
  }

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
    recommendedSpaces: recommendedSpaces.slice(0, 4),
    cautionSpaces: cautionSpaces.slice(0, 4),
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
        <SummarySection
          title="Recommended Spaces"
          lines={summary.recommendedSpaces}
        />
        <SummarySection title="Caution Spaces" lines={summary.cautionSpaces} />
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

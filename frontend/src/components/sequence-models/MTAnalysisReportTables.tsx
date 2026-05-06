import type { MaskedTransformerAnalysisReport } from "@/api/sequenceModels";

type Indicator = "good" | "warn" | "bad" | "none";

const POSITIVE_METRICS = new Set([
  "same_human_label",
  "exact_human_label_set",
  "similar_duration",
]);
const NEGATIVE_METRICS = new Set([
  "same_event",
  "same_region",
  "adjacent_1s",
  "nearby_5s",
  "without_human_label",
  "low_event_overlap",
]);

export function classifyAnalysisMetric(key: string, value: unknown): Indicator {
  if (typeof value !== "number" || !Number.isFinite(value)) return "none";
  if (POSITIVE_METRICS.has(key)) {
    if (value >= 0.5) return "good";
    if (value >= 0.25) return "warn";
    return "bad";
  }
  if (NEGATIVE_METRICS.has(key)) {
    if (value <= 0.25) return "good";
    if (value <= 0.5) return "warn";
    return "bad";
  }
  return "none";
}

function indicatorClass(indicator: Indicator): string {
  if (indicator === "good") return "bg-emerald-50 text-emerald-900";
  if (indicator === "warn") return "bg-amber-50 text-amber-900";
  if (indicator === "bad") return "bg-red-50 text-red-900";
  return "";
}

function worstIndicator(values: Indicator[]): Indicator {
  if (values.includes("bad")) return "bad";
  if (values.includes("warn")) return "warn";
  if (values.includes("good")) return "good";
  return "none";
}

function classifyGeometryBand(kind: "mean" | "rank", value: unknown): Indicator {
  if (kind === "mean") {
    if (value === "good") return "good";
    if (value === "okay") return "warn";
    if (value === "suspicious" || value === "collapse_risk") return "bad";
    return "none";
  }
  if (value === "broad" || value === "plausible") return "good";
  if (value === "weak") return "warn";
  if (value === "severe_collapse") return "bad";
  return "none";
}

function formatValue(value: unknown): string {
  if (value == null) return "-";
  if (typeof value === "number") return Number.isInteger(value) ? String(value) : value.toFixed(3);
  if (typeof value === "boolean") return value ? "true" : "false";
  if (Array.isArray(value)) return value.join(", ");
  if (typeof value === "object") return JSON.stringify(value);
  return String(value);
}

export function MTAnalysisReportTables({
  report,
}: {
  report: MaskedTransformerAnalysisReport;
}) {
  const metricRows = Object.entries(report.results).flatMap(([mode, variants]) =>
    Object.entries(variants).flatMap(([variant, metrics]) =>
      Object.entries(metrics)
        .filter(([, value]) => typeof value !== "object" || value == null)
        .map(([metric, value]) => ({ mode, variant, metric, value })),
    ),
  );
  const eventRows = report.event_level_results
    ? Object.entries(report.event_level_results).flatMap(([mode, variants]) =>
        Object.entries(variants).flatMap(([variant, metrics]) =>
          Object.entries(metrics)
            .filter(([, value]) => typeof value !== "object" || value == null)
            .map(([metric, value]) => ({ mode, variant, metric, value })),
        ),
      )
    : [];
  const geometryRows = report.geometry_report
    ? Object.entries(report.geometry_report.spaces).map(([space, row]) => ({
        space,
        ...row,
      }))
    : [];

  return (
    <div className="space-y-4" data-testid="mt-analysis-report-tables">
      <KeyValueTable title="Report" rows={report.job} />
      <KeyValueTable title="Label Coverage" rows={report.label_coverage} />
      <MetricTable title="Aggregate Retrieval Metrics" rows={metricRows} />
      {eventRows.length > 0 ? (
        <MetricTable title="Event-Level Metrics" rows={eventRows} />
      ) : null}
      {geometryRows.length > 0 ? <GeometryTable rows={geometryRows} /> : null}
      <QueryTable title="Representative Good Queries" rows={report.representative_good_queries} />
      <QueryTable title="Representative Risky Queries" rows={report.representative_risky_queries} />
    </div>
  );
}

function KeyValueTable({
  title,
  rows,
}: {
  title: string;
  rows: Record<string, unknown>;
}) {
  return (
    <div className="rounded-md border">
      <div className="border-b px-3 py-2 text-sm font-semibold">{title}</div>
      <table className="w-full text-xs">
        <tbody>
          {Object.entries(rows).map(([key, value]) => (
            <tr key={key} className="border-b last:border-0">
              <td className="w-64 px-3 py-1 font-medium">{key}</td>
              <td className="px-3 py-1">{formatValue(value)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function MetricTable({
  title,
  rows,
}: {
  title: string;
  rows: { mode: string; variant: string; metric: string; value: unknown }[];
}) {
  return (
    <div className="rounded-md border">
      <div className="border-b px-3 py-2 text-sm font-semibold">{title}</div>
      <table className="w-full text-xs">
        <thead>
          <tr className="border-b bg-muted/50">
            <th className="px-3 py-1 text-left">Mode</th>
            <th className="px-3 py-1 text-left">Variant</th>
            <th className="px-3 py-1 text-left">Metric</th>
            <th className="px-3 py-1 text-left">Value</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => (
            <tr
              key={`${row.mode}:${row.variant}:${row.metric}`}
              className={`border-b ${indicatorClass(classifyAnalysisMetric(row.metric, row.value))}`}
            >
              <td className="px-3 py-1">{row.mode}</td>
              <td className="px-3 py-1">{row.variant}</td>
              <td className="px-3 py-1">{row.metric}</td>
              <td className="px-3 py-1">{formatValue(row.value)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function GeometryTable({ rows }: { rows: Record<string, unknown>[] }) {
  return (
    <div className="rounded-md border" data-testid="mt-analysis-geometry-table">
      <div className="border-b px-3 py-2 text-sm font-semibold">
        Geometry Diagnostics
      </div>
      <table className="w-full text-xs">
        <thead>
          <tr className="border-b bg-muted/50">
            <th className="px-3 py-1 text-left">Space</th>
            <th className="px-3 py-1 text-left">Available</th>
            <th className="px-3 py-1 text-left">Mean Band</th>
            <th className="px-3 py-1 text-left">Rank Band</th>
            <th className="px-3 py-1 text-left">Warnings</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => {
            const indicator =
              row.available === false
                ? "warn"
                : worstIndicator([
                    classifyGeometryBand("mean", row.mean_vector_band),
                    classifyGeometryBand("rank", row.effective_rank_band),
                  ]);
            return (
              <tr key={String(row.space)} className={`border-b ${indicatorClass(indicator)}`}>
                <td className="px-3 py-1">{formatValue(row.space)}</td>
                <td className="px-3 py-1">{formatValue(row.available)}</td>
                <td className="px-3 py-1">{formatValue(row.mean_vector_band)}</td>
                <td className="px-3 py-1">{formatValue(row.effective_rank_band)}</td>
                <td className="px-3 py-1">{formatValue(row.warnings)}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

function QueryTable({
  title,
  rows,
}: {
  title: string;
  rows: Record<string, unknown>[];
}) {
  return (
    <div className="rounded-md border">
      <div className="border-b px-3 py-2 text-sm font-semibold">{title}</div>
      <table className="w-full text-xs">
        <thead>
          <tr className="border-b bg-muted/50">
            <th className="px-3 py-1 text-left">Query</th>
            <th className="px-3 py-1 text-left">Label</th>
            <th className="px-3 py-1 text-left">Verdict</th>
            <th className="px-3 py-1 text-left">Same Label</th>
            <th className="px-3 py-1 text-left">Adjacent</th>
          </tr>
        </thead>
        <tbody>
          {rows.length === 0 ? (
            <tr>
              <td colSpan={5} className="px-3 py-3 text-center text-muted-foreground">
                No rows.
              </td>
            </tr>
          ) : null}
          {rows.map((row, index) => (
            <tr key={index} className="border-b">
              <td className="px-3 py-1">{formatValue(row.query_idx)}</td>
              <td className="px-3 py-1">{formatValue(row.query_human_types)}</td>
              <td className="px-3 py-1">{formatValue(row.verdict)}</td>
              <td className="px-3 py-1">{formatValue(row.same_human_label_rate)}</td>
              <td className="px-3 py-1">{formatValue(row.adjacent_1s_rate)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

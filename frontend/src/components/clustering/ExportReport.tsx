import { useCallback, useState } from "react";
import { Button } from "@/components/ui/button";
import { Download } from "lucide-react";
import { fetchVisualization, fetchMetrics, fetchDendrogram } from "@/api/client";
import type { VisualizationData, ClusteringMetrics, DendrogramData } from "@/api/types";
import { shortId } from "@/utils/format";
import { showMsg } from "@/components/shared/MessageToast";

const PALETTE = [
  "#3a86ff", "#e63946", "#2a9d8f", "#e9c46a", "#264653",
  "#f4a261", "#6c5ce7", "#00b894", "#fd79a8", "#636e72",
  "#d63031", "#74b9ff", "#a29bfe", "#ffeaa7", "#55efc4",
];

function buildUmapTraces(viz: VisualizationData) {
  const groups = new Map<number, number[]>();
  for (let i = 0; i < viz.cluster_label.length; i++) {
    const label = viz.cluster_label[i];
    if (!groups.has(label)) groups.set(label, []);
    groups.get(label)!.push(i);
  }

  const sortedLabels = [...groups.keys()].sort((a, b) => a - b);
  return sortedLabels.map((label) => {
    const indices = groups.get(label)!;
    const isNoise = label === -1;
    return {
      x: indices.map((i) => viz.x[i]),
      y: indices.map((i) => viz.y[i]),
      mode: "markers",
      type: "scatter",
      name: isNoise ? "Noise" : `Cluster ${label}`,
      marker: {
        color: isNoise ? "#b2bec3" : PALETTE[label % PALETTE.length],
        size: isNoise ? 4 : 7,
        opacity: isNoise ? 0.4 : 0.8,
      },
      text: indices.map((i) => {
        const fn = viz.audio_filename[i];
        return fn ? fn.replace(/\.[^.]+$/, "") : `ES: ${shortId(viz.embedding_set_id[i])}`;
      }),
      hoverinfo: "text+name",
      customdata: indices.map((i) => `${viz.audio_filename[i]} (row ${viz.embedding_row_index[i]})`),
    };
  });
}

function buildLabelPlotData(metrics: ClusteringMetrics) {
  const cm = metrics.confusion_matrix as Record<string, Record<string, number>> | undefined;
  if (!cm || Object.keys(cm).length === 0) return null;

  const categories = Object.keys(cm).sort();
  const clusterSet = new Set<string>();
  for (const counts of Object.values(cm)) {
    for (const cl of Object.keys(counts)) clusterSet.add(cl);
  }
  const clusterLabels = [...clusterSet].sort((a, b) => Number(a) - Number(b));

  const xs: string[] = [];
  const ys: string[] = [];
  const sizes: number[] = [];
  const counts: number[] = [];

  for (const cat of categories) {
    for (const cl of clusterLabels) {
      const count = cm[cat]?.[cl] ?? 0;
      if (count === 0) continue;
      xs.push(cat);
      ys.push(cl);
      sizes.push(count);
      counts.push(count);
    }
  }

  if (sizes.length === 0) return null;

  const catTotals = new Map<string, number>();
  for (let i = 0; i < xs.length; i++) {
    catTotals.set(xs[i], (catTotals.get(xs[i]) ?? 0) + counts[i]);
  }

  const logProps = counts.map((c, i) =>
    Math.log1p(c / (catTotals.get(xs[i]) ?? 1)),
  );

  const mean = logProps.reduce((a, b) => a + b, 0) / logProps.length;
  const std =
    Math.sqrt(logProps.reduce((a, b) => a + (b - mean) ** 2, 0) / logProps.length) || 1;
  const zScores = logProps.map((v) => (v - mean) / std);

  const maxCount = Math.max(...sizes);
  const scaledSizes = sizes.map((s) => 6 + (s / maxCount) * 34);

  const trace = {
    x: xs,
    y: ys,
    mode: "markers",
    type: "scatter",
    marker: {
      size: scaledSizes,
      color: zScores,
      colorscale: [[0, "#f2f0f7"], [0.5, "#9e9ac8"], [1, "#3f007d"]],
      showscale: true,
      colorbar: { title: { text: "Z-score" }, thickness: 12, len: 0.6 },
    },
    text: counts.map((c, i) => `Category: ${xs[i]}<br>Cluster: ${ys[i]}<br>Count: ${c}`),
    hoverinfo: "text",
  };

  const layout = {
    xaxis: { title: { text: "Category" }, type: "category", tickangle: -45 },
    yaxis: { title: { text: "Cluster" }, type: "category", autorange: "reversed" },
    margin: { l: 60, r: 40, t: 20, b: 120 },
    height: Math.max(300, clusterLabels.length * 28 + 160),
    hovermode: "closest",
  };

  return { trace, layout };
}

function buildDendrogramPlotData(data: DendrogramData) {
  const { categories, cluster_labels, values, raw_counts, row_dendrogram, col_dendrogram } = data;

  const nRows = cluster_labels.length;
  const nCols = categories.length;

  // Column dendrogram traces (top)
  const colDendroTraces = col_dendrogram.icoord.map((ic, idx) => ({
    x: ic,
    y: col_dendrogram.dcoord[idx],
    type: "scatter",
    mode: "lines",
    line: { color: "#64748b", width: 1 },
    xaxis: "x",
    yaxis: "y2",
    hoverinfo: "skip",
    showlegend: false,
  }));

  // Row dendrogram traces (left) — rotated
  const rowDendroTraces = row_dendrogram.icoord.map((ic, idx) => ({
    x: row_dendrogram.dcoord[idx].map((d: number) => -d),
    y: ic,
    type: "scatter",
    mode: "lines",
    line: { color: "#64748b", width: 1 },
    xaxis: "x3",
    yaxis: "y3",
    hoverinfo: "skip",
    showlegend: false,
  }));

  // Heatmap trace
  const heatmapTrace = {
    z: values,
    x: categories,
    y: cluster_labels,
    type: "heatmap",
    colorscale: [
      [0, "#f0f4ff"],
      [0.2, "#c7d7fe"],
      [0.4, "#818cf8"],
      [0.6, "#6366f1"],
      [0.8, "#4338ca"],
      [1, "#312e81"],
    ],
    xaxis: "x4",
    yaxis: "y4",
    customdata: raw_counts,
    hovertemplate:
      "Category: %{x}<br>Cluster: %{y}<br>Count: %{customdata}<br>Proportion: %{z:.3f}<extra></extra>",
    showscale: true,
    colorbar: {
      title: { text: "Proportion" },
      thickness: 12,
      len: 0.55,
      y: 0.25,
      yanchor: "middle",
    },
    zmin: 0,
    zmax: 1,
  };

  // Compute axis ranges
  const colIAll = col_dendrogram.icoord.flat();
  const colDAll = col_dendrogram.dcoord.flat();
  const rowIAll = row_dendrogram.icoord.flat();
  const rowDAll = row_dendrogram.dcoord.flat();

  const colXRange = [Math.min(...colIAll) - 5, Math.max(...colIAll) + 5];
  const colYRange = [0, Math.max(...colDAll) * 1.05];
  const rowYRange = [Math.min(...rowIAll) - 5, Math.max(...rowIAll) + 5];
  const rowXRange = [-Math.max(...rowDAll) * 1.05, 0];

  const layout = {
    xaxis: {
      domain: [0.15, 0.92],
      range: colXRange,
      showticklabels: false,
      showgrid: false,
      zeroline: false,
      showline: false,
    },
    yaxis2: {
      domain: [0.82, 1],
      range: colYRange,
      showticklabels: false,
      showgrid: false,
      zeroline: false,
      showline: false,
    },
    xaxis3: {
      domain: [0, 0.13],
      range: rowXRange,
      showticklabels: false,
      showgrid: false,
      zeroline: false,
      showline: false,
    },
    yaxis3: {
      domain: [0, 0.8],
      range: rowYRange,
      showticklabels: false,
      showgrid: false,
      zeroline: false,
      showline: false,
    },
    xaxis4: {
      domain: [0.15, 0.92],
      tickangle: -45,
    },
    yaxis4: {
      domain: [0, 0.8],
      autorange: "reversed",
    },
    height: Math.max(350, nRows * 28 + 200),
    margin: { l: 100, r: 60, t: 30, b: Math.min(150, nCols * 10 + 60) },
    hovermode: "closest",
    showlegend: false,
  };

  return {
    traces: [...colDendroTraces, ...rowDendroTraces, heatmapTrace],
    layout,
  };
}

interface ExportReportProps {
  jobId: string;
}

export function ExportReport({ jobId }: ExportReportProps) {
  const [exporting, setExporting] = useState(false);

  const handleExport = useCallback(async () => {
    setExporting(true);
    try {
      const [viz, metrics, dendrogramData] = await Promise.all([
        fetchVisualization(jobId),
        fetchMetrics(jobId),
        fetchDendrogram(jobId).catch(() => null),
      ]);

      // UMAP traces
      const umapTraces = buildUmapTraces(viz);
      const umapLayout = {
        xaxis: { title: "UMAP 1" },
        yaxis: { title: "UMAP 2" },
        legend: { x: 1.02, y: 1 },
        margin: { l: 50, r: 120, t: 20, b: 50 },
        hovermode: "closest",
        height: 600,
      };

      // Label distribution plot
      const labelPlot = buildLabelPlotData(metrics);

      // Dendrogram heatmap
      const dendrogramPlot = dendrogramData ? buildDendrogramPlotData(dendrogramData) : null;

      // Build metrics table HTML
      const metricRows = Object.entries(metrics)
        .filter(([, v]) => v != null && typeof v !== "object")
        .map(
          ([k, v]) =>
            `<tr><td style="padding:4px 12px;border-bottom:1px solid #eee">${k}</td><td style="padding:4px 12px;border-bottom:1px solid #eee;font-family:monospace">${typeof v === "number" ? v.toFixed(4) : v}</td></tr>`,
        )
        .join("");

      const categoryMetrics = metrics.category_metrics as Record<string, Record<string, number>> | undefined;
      let categoryHtml = "";
      if (categoryMetrics && Object.keys(categoryMetrics).length > 0) {
        categoryHtml = `<h3>Category Metrics</h3><table style="border-collapse:collapse;margin:10px 0"><tr><th style="padding:4px 12px;text-align:left;border-bottom:2px solid #ccc">Category</th><th style="padding:4px 12px;text-align:left;border-bottom:2px solid #ccc">Purity</th><th style="padding:4px 12px;text-align:left;border-bottom:2px solid #ccc">Count</th></tr>`;
        for (const [cat, vals] of Object.entries(categoryMetrics)) {
          categoryHtml += `<tr><td style="padding:4px 12px;border-bottom:1px solid #eee">${cat}</td><td style="padding:4px 12px;border-bottom:1px solid #eee;font-family:monospace">${vals.purity?.toFixed(4) ?? "\u2014"}</td><td style="padding:4px 12px;border-bottom:1px solid #eee;font-family:monospace">${vals.count ?? "\u2014"}</td></tr>`;
        }
        categoryHtml += "</table>";
      }

      // Build script sections for each plot
      let plotScripts = `
var umapTraces=${JSON.stringify(umapTraces)};
var umapLayout=${JSON.stringify(umapLayout)};
Plotly.newPlot('umap-plot',umapTraces,umapLayout);
document.getElementById('umap-plot').on('plotly_click',function(d){
  var pt=d.points[0];if(!pt||!pt.customdata)return;
  var tip=document.getElementById('click-tip');
  tip.textContent=pt.customdata;
  tip.style.display='block';
  tip.style.left=(d.event.clientX+10)+'px';
  tip.style.top=(d.event.clientY-30)+'px';
  setTimeout(function(){tip.style.display='none'},3000);
});`;

      if (labelPlot) {
        plotScripts += `
var labelTrace=${JSON.stringify(labelPlot.trace)};
var labelLayout=${JSON.stringify(labelPlot.layout)};
Plotly.newPlot('label-plot',[labelTrace],labelLayout);`;
      }

      if (dendrogramPlot) {
        plotScripts += `
var dendroTraces=${JSON.stringify(dendrogramPlot.traces)};
var dendroLayout=${JSON.stringify(dendrogramPlot.layout)};
Plotly.newPlot('dendro-plot',dendroTraces,dendroLayout);`;
      }

      const sid = shortId(jobId);
      const html = `<!DOCTYPE html>
<html lang="en">
<head><meta charset="UTF-8"><title>Cluster Report - Job ${sid}</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"><\/script>
<style>body{font-family:system-ui,sans-serif;max-width:1200px;margin:0 auto;padding:20px}h1{color:#1e293b}h3{color:#334155;margin-top:24px}table{border-collapse:collapse}th{text-align:left;border-bottom:2px solid #ccc;padding:4px 12px}.tooltip{position:fixed;background:#1e293b;color:#fff;padding:6px 12px;border-radius:6px;font-size:13px;pointer-events:none;z-index:1000;display:none}.section{margin-top:32px}</style>
</head>
<body>
<h1>Cluster Report &mdash; Job ${sid}</h1>

<div class="section">
<h3>UMAP Projection</h3>
<div id="umap-plot" style="width:100%;height:600px"></div>
<div class="tooltip" id="click-tip"></div>
</div>

${labelPlot ? `<div class="section">
<h3>Label Distribution</h3>
<div id="label-plot" style="width:100%"></div>
</div>` : ""}

${dendrogramPlot ? `<div class="section">
<h3>Cluster &times; Category Dendrogram</h3>
<div id="dendro-plot" style="width:100%"></div>
</div>` : ""}

<div class="section">
<h3>Metrics</h3>
<table><tr><th>Metric</th><th>Value</th></tr>${metricRows}</table>
${categoryHtml}
</div>

<p style="color:#94a3b8;font-size:12px;margin-top:30px">Generated ${new Date().toLocaleString()}</p>
<script>
${plotScripts}
<\/script>
</body></html>`;

      const blob = new Blob([html], { type: "text/html" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `cluster_report_${sid}.html`;
      a.click();
      URL.revokeObjectURL(url);
    } catch (e) {
      showMsg("error", `Export failed: ${e instanceof Error ? e.message : "Unknown error"}`);
    } finally {
      setExporting(false);
    }
  }, [jobId]);

  return (
    <Button variant="outline" size="sm" onClick={handleExport} disabled={exporting}>
      <Download className="h-3.5 w-3.5 mr-1" />
      {exporting ? "Exporting..." : "Export Report"}
    </Button>
  );
}

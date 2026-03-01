import { useCallback, useState } from "react";
import { Button } from "@/components/ui/button";
import { Download } from "lucide-react";
import { fetchVisualization, fetchMetrics } from "@/api/client";
import { shortId } from "@/utils/format";
import { showMsg } from "@/components/shared/MessageToast";

const PALETTE = [
  "#3a86ff", "#e63946", "#2a9d8f", "#e9c46a", "#264653",
  "#f4a261", "#6c5ce7", "#00b894", "#fd79a8", "#636e72",
  "#d63031", "#74b9ff", "#a29bfe", "#ffeaa7", "#55efc4",
];

interface ExportReportProps {
  jobId: string;
}

export function ExportReport({ jobId }: ExportReportProps) {
  const [exporting, setExporting] = useState(false);

  const handleExport = useCallback(async () => {
    setExporting(true);
    try {
      const [viz, metrics] = await Promise.all([
        fetchVisualization(jobId),
        fetchMetrics(jobId),
      ]);

      // Build traces
      const groups = new Map<number, number[]>();
      for (let i = 0; i < viz.cluster_label.length; i++) {
        const label = viz.cluster_label[i];
        if (!groups.has(label)) groups.set(label, []);
        groups.get(label)!.push(i);
      }

      const sortedLabels = [...groups.keys()].sort((a, b) => a - b);
      const traces = sortedLabels.map((label) => {
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
            const name = fn ? fn.replace(/\.[^.]+$/, "") : `ES: ${shortId(viz.embedding_set_id[i])}`;
            return name;
          }),
          hoverinfo: "text+name",
          customdata: indices.map((i) => `${viz.audio_filename[i]} (row ${viz.embedding_row_index[i]})`),
        };
      });

      const layout = {
        xaxis: { title: "UMAP 1" },
        yaxis: { title: "UMAP 2" },
        legend: { x: 1.02, y: 1 },
        margin: { l: 50, r: 120, t: 20, b: 50 },
        hovermode: "closest",
      };

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
          categoryHtml += `<tr><td style="padding:4px 12px;border-bottom:1px solid #eee">${cat}</td><td style="padding:4px 12px;border-bottom:1px solid #eee;font-family:monospace">${vals.purity?.toFixed(4) ?? "—"}</td><td style="padding:4px 12px;border-bottom:1px solid #eee;font-family:monospace">${vals.count ?? "—"}</td></tr>`;
        }
        categoryHtml += "</table>";
      }

      const sid = shortId(jobId);
      const html = `<!DOCTYPE html>
<html lang="en">
<head><meta charset="UTF-8"><title>Cluster Report - Job ${sid}</title>
<script src="https://cdn.plot.ly/plotly-basic-2.35.2.min.js"><\/script>
<style>body{font-family:system-ui,sans-serif;max-width:1200px;margin:0 auto;padding:20px}h1{color:#1e293b}table{border-collapse:collapse}th{text-align:left;border-bottom:2px solid #ccc;padding:4px 12px}.tooltip{position:fixed;background:#1e293b;color:#fff;padding:6px 12px;border-radius:6px;font-size:13px;pointer-events:none;z-index:1000;display:none}</style>
</head>
<body>
<h1>Cluster Report &mdash; Job ${sid}</h1>
<div id="umap-plot" style="width:100%;height:600px"></div>
<div class="tooltip" id="click-tip"></div>
<h3>Metrics</h3>
<table><tr><th>Metric</th><th>Value</th></tr>${metricRows}</table>
${categoryHtml}
<p style="color:#94a3b8;font-size:12px;margin-top:30px">Generated ${new Date().toLocaleString()}</p>
<script>
var traces=${JSON.stringify(traces)};
var layout=${JSON.stringify(layout)};
layout.height=600;
Plotly.newPlot('umap-plot',traces,layout);
document.getElementById('umap-plot').on('plotly_click',function(d){
  var pt=d.points[0];if(!pt||!pt.customdata)return;
  var tip=document.getElementById('click-tip');
  tip.textContent=pt.customdata;
  tip.style.display='block';
  tip.style.left=(d.event.clientX+10)+'px';
  tip.style.top=(d.event.clientY-30)+'px';
  setTimeout(function(){tip.style.display='none'},3000);
});
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

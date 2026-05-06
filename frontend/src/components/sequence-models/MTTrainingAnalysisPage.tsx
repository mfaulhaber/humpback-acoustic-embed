import { Link, useParams } from "react-router-dom";
import { useLatestMaskedTransformerAnalysis } from "@/api/sequenceModels";
import { MTAnalysisReportTables } from "./MTAnalysisReportTables";

export function MTTrainingAnalysisPage() {
  const { jobId = "" } = useParams<{ jobId: string }>();
  const { data, isLoading, error } = useLatestMaskedTransformerAnalysis(jobId);

  return (
    <div className="space-y-4 p-2" data-testid="mt-training-analysis-page">
      <div className="flex items-center justify-between rounded-md border p-4">
        <h1 className="text-lg font-semibold">MT Training Analysis</h1>
        <Link
          className="text-xs underline text-muted-foreground"
          to={`/app/sequence-models/mt-training/${jobId}`}
        >
          Back
        </Link>
      </div>
      {isLoading ? (
        <div className="text-sm text-muted-foreground">Loading...</div>
      ) : error ? (
        <div className="rounded-md border border-amber-200 bg-amber-50 p-3 text-sm text-amber-900">
          No analysis report has been generated for this job.
        </div>
      ) : data ? (
        <MTAnalysisReportTables report={data} />
      ) : null}
    </div>
  );
}

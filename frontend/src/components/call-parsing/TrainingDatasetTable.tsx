import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  useSegmentationTrainingDatasets,
  useCreateSegmentationTrainingJob,
  useSegmentationJobsWithCorrectionCounts,
} from "@/hooks/queries/useCallParsing";
import { toast } from "@/components/ui/use-toast";

export function TrainingDatasetTable() {
  const { data: datasets = [] } = useSegmentationTrainingDatasets();
  const trainMutation = useCreateSegmentationTrainingJob();
  const { data: correctionJobs = [] } =
    useSegmentationJobsWithCorrectionCounts();
  const jobsWithNewCorrections = correctionJobs.filter(
    (j) => j.has_new_corrections,
  );

  const handleTrain = (datasetId: string, datasetName: string) => {
    if (
      !confirm(
        `Train a new segmentation model from dataset "${datasetName}"?`,
      )
    )
      return;
    trainMutation.mutate(
      { training_dataset_id: datasetId },
      {
        onSuccess: () => {
          toast({
            title: "Training job queued",
            description:
              "The model will train in the background. Check the models table for results.",
          });
        },
        onError: (err) => {
          toast({
            title: "Failed to start training",
            description: (err as Error).message,
            variant: "destructive",
          });
        },
      },
    );
  };

  return (
    <div className="border rounded-md">
      <div className="flex items-center justify-between px-4 py-3 border-b">
        <div className="flex items-center gap-2">
          <h3 className="text-sm font-semibold">Training Datasets</h3>
          <Badge variant="secondary">{datasets.length}</Badge>
        </div>
      </div>

      {jobsWithNewCorrections.length > 0 && (
        <div className="px-4 py-2 bg-amber-50 dark:bg-amber-950/30 border-b text-sm text-amber-700 dark:text-amber-400">
          New corrections available from{" "}
          {jobsWithNewCorrections.length} segmentation job
          {jobsWithNewCorrections.length !== 1 ? "s" : ""}. Create a
          new dataset above to include them.
        </div>
      )}

      {datasets.length === 0 ? (
        <div className="px-4 py-6 text-center text-sm text-muted-foreground">
          No training datasets yet. Select segmentation jobs above to create
          one.
        </div>
      ) : (
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b bg-muted/50">
              <th className="px-3 py-2 text-left font-medium">Name</th>
              <th className="px-3 py-2 text-right font-medium">Samples</th>
              <th className="px-3 py-2 text-right font-medium">Source Jobs</th>
              <th className="px-3 py-2 text-left font-medium">Created</th>
              <th className="px-3 py-2 text-left font-medium" />
            </tr>
          </thead>
          <tbody>
            {datasets.map((ds) => (
              <tr key={ds.id} className="border-b hover:bg-muted/30">
                <td className="px-3 py-2 font-medium">{ds.name}</td>
                <td className="px-3 py-2 text-right font-mono text-xs">
                  {ds.sample_count}
                </td>
                <td className="px-3 py-2 text-right font-mono text-xs">
                  {ds.source_job_count}
                </td>
                <td className="px-3 py-2 text-xs text-muted-foreground whitespace-nowrap">
                  {new Date(ds.created_at).toLocaleDateString()}
                </td>
                <td className="px-3 py-2 text-right">
                  <Button
                    variant="outline"
                    size="sm"
                    disabled={trainMutation.isPending}
                    onClick={() => handleTrain(ds.id, ds.name)}
                  >
                    Train
                  </Button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { DeleteConfirmButton } from "@/components/shared/DeleteConfirmationDialog";
import {
  useSegmentationTrainingDatasets,
  useCreateSegmentationTrainingJob,
  useDeleteSegmentationTrainingDataset,
  useSegmentationJobsWithCorrectionCounts,
} from "@/hooks/queries/useCallParsing";
import { toast } from "@/components/ui/use-toast";

export function TrainingDatasetTable() {
  const { data: datasets = [] } = useSegmentationTrainingDatasets();
  const trainMutation = useCreateSegmentationTrainingJob();
  const deleteMutation = useDeleteSegmentationTrainingDataset();
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

  const handleDelete = async (datasetId: string) => {
    try {
      await deleteMutation.mutateAsync(datasetId);
      toast({
        title: "Training dataset deleted",
        description: "The dataset and its saved samples were removed.",
      });
    } catch (err) {
      toast({
        title: "Cannot delete training dataset",
        description: (err as Error).message,
        variant: "destructive",
      });
      throw err;
    }
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
                  <div className="flex justify-end gap-2">
                    <Button
                      variant="outline"
                      size="sm"
                      disabled={
                        trainMutation.isPending || deleteMutation.isPending
                      }
                      onClick={() => handleTrain(ds.id, ds.name)}
                    >
                      Train
                    </Button>
                    <DeleteConfirmButton
                      resourceType="training dataset"
                      resourceName={ds.name}
                      consequence="The dataset and its saved samples will be removed. Existing models and completed or failed training jobs will remain."
                      onConfirm={() => handleDelete(ds.id)}
                      isPending={deleteMutation.isPending}
                    >
                      Delete
                    </DeleteConfirmButton>
                  </div>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}

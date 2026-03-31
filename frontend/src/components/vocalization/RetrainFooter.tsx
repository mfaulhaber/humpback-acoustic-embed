import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Loader2, RefreshCw, CheckCircle } from "lucide-react";
import {
  useVocClassifierModels,
  useVocClassifierInferenceJob,
  useVocModelTrainingSource,
  useCreateVocClassifierTrainingJob,
  useVocClassifierTrainingJob,
  useActivateVocClassifierModel,
  useExtendTrainingDataset,
} from "@/hooks/queries/useVocalization";
import type { LabelingSource } from "@/api/types";

interface Props {
  source: LabelingSource;
  inferenceJobId: string;
  labelCount: number;
}

export function RetrainFooter({ source, inferenceJobId, labelCount }: Props) {
  const { data: models = [] } = useVocClassifierModels();
  const { data: inferenceJob } = useVocClassifierInferenceJob(inferenceJobId);

  // Use the model from the current inference job, falling back to active model
  const inferenceModelId = inferenceJob?.vocalization_model_id ?? null;
  const activeModel = models.find((m) => m.is_active);
  const retrainModel =
    models.find((m) => m.id === inferenceModelId) ?? activeModel ?? null;

  const { data: trainingSource } = useVocModelTrainingSource(
    retrainModel?.id ?? null,
  );

  const createTraining = useCreateVocClassifierTrainingJob();
  const extendDataset = useExtendTrainingDataset();
  const activateMut = useActivateVocClassifierModel();
  const [trainingJobId, setTrainingJobId] = useState<string | null>(null);
  const [extending, setExtending] = useState(false);
  const { data: trainingJob } = useVocClassifierTrainingJob(trainingJobId);

  const isTraining =
    trainingJob?.status === "queued" || trainingJob?.status === "running";
  const isComplete = trainingJob?.status === "complete";

  async function handleRetrain() {
    if (!retrainModel) return;

    const datasetId = retrainModel.training_dataset_id;

    if (datasetId) {
      // New flow: extend dataset with current detection job, then retrain
      setExtending(true);
      try {
        if (source.type === "detection_job") {
          await extendDataset.mutateAsync({
            datasetId,
            body: { detection_job_ids: [source.jobId] },
          });
        }
        createTraining.mutate(
          { training_dataset_id: datasetId },
          { onSuccess: (job) => setTrainingJobId(job.id) },
        );
      } finally {
        setExtending(false);
      }
    } else {
      // Legacy flow: build source_config from scratch
      const existingConfig = trainingSource?.source_config ?? {
        embedding_set_ids: [] as string[],
        detection_job_ids: [] as string[],
      };

      const detJobIds: string[] = [
        ...(existingConfig.detection_job_ids ?? []),
      ];
      if (
        source.type === "detection_job" &&
        !detJobIds.includes(source.jobId)
      ) {
        detJobIds.push(source.jobId);
      }

      createTraining.mutate(
        {
          source_config: {
            embedding_set_ids: existingConfig.embedding_set_ids ?? [],
            detection_job_ids: detJobIds,
          },
          parameters:
            (trainingSource?.parameters as Record<string, unknown>) ??
            undefined,
        },
        {
          onSuccess: (job) => setTrainingJobId(job.id),
        },
      );
    }
  }

  if (labelCount === 0 && !trainingJobId) return null;

  return (
    <div className="sticky bottom-0 bg-background border-t px-4 py-3 flex items-center justify-between gap-4">
      <div className="text-sm text-muted-foreground">
        {labelCount > 0 && (
          <>
            <span className="font-medium text-foreground">{labelCount}</span>{" "}
            new label{labelCount !== 1 ? "s" : ""} this session
          </>
        )}
      </div>

      <div className="flex items-center gap-2">
        {isComplete && trainingJob?.vocalization_model_id && (
          <>
            <Badge variant="outline" className="text-xs text-green-700">
              <CheckCircle className="h-3 w-3 mr-1" />
              Training complete
            </Badge>
            <Button
              size="sm"
              variant="outline"
              onClick={() => {
                if (trainingJob.vocalization_model_id) {
                  activateMut.mutate(trainingJob.vocalization_model_id);
                }
              }}
              disabled={activateMut.isPending}
            >
              Activate new model
            </Button>
          </>
        )}

        {isTraining && (
          <div className="flex items-center gap-2 text-sm text-muted-foreground">
            <Loader2 className="h-4 w-4 animate-spin" />
            Training...
          </div>
        )}

        {!isTraining && !isComplete && (
          <Button
            size="sm"
            onClick={handleRetrain}
            disabled={
              extending ||
              createTraining.isPending ||
              labelCount === 0 ||
              !retrainModel
            }
          >
            {extending || createTraining.isPending ? (
              <Loader2 className="h-3.5 w-3.5 mr-1 animate-spin" />
            ) : (
              <RefreshCw className="h-3.5 w-3.5 mr-1" />
            )}
            {extending ? "Extending dataset..." : "Retrain Model"}
          </Button>
        )}
      </div>
    </div>
  );
}

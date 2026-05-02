import { Routes, Route, Navigate } from "react-router-dom";
import { Toaster } from "@/components/ui/toaster";
import { AppShell } from "@/components/layout/AppShell";
import { AdminTab } from "@/components/admin/AdminTab";
import { TrainingTab } from "@/components/classifier/TrainingTab";
import { HydrophoneTab } from "@/components/classifier/HydrophoneTab";
import { LabelingTab } from "@/components/classifier/LabelingTab";
import { ClassifierTimeline } from "@/components/timeline/ClassifierTimeline";
import { EmbeddingsPage } from "@/components/classifier/EmbeddingsPage";
import { TuningTab } from "@/components/classifier/TuningTab";
import { VocalizationTrainingTab } from "@/components/vocalization/VocalizationTrainingTab";
import { VocalizationLabelingTab } from "@/components/vocalization/VocalizationLabelingTab";
import { TrainingDataView } from "@/components/vocalization/TrainingDataView";
import { VocalizationClusteringPage } from "@/components/vocalization/VocalizationClusteringPage";
import { VocalizationClusteringDetail } from "@/components/vocalization/VocalizationClusteringDetail";
import { DetectionPage } from "@/components/call-parsing/DetectionPage";
import { SegmentPage } from "@/components/call-parsing/SegmentPage";
import { SegmentTrainingPage } from "@/components/call-parsing/SegmentTrainingPage";
import { ClassifyPage } from "@/components/call-parsing/ClassifyPage";
import { ClassifyTrainingPage } from "@/components/call-parsing/ClassifyTrainingPage";
import { WindowClassifyPage } from "@/components/call-parsing/WindowClassifyPage";
import { RegionDetectionTimeline } from "@/components/call-parsing/RegionDetectionTimeline";
import { ContinuousEmbeddingJobsPage } from "@/components/sequence-models/ContinuousEmbeddingJobsPage";
import { ContinuousEmbeddingDetailPage } from "@/components/sequence-models/ContinuousEmbeddingDetailPage";
import { HMMSequenceJobsPage } from "@/components/sequence-models/HMMSequenceJobsPage";
import { HMMSequenceDetailPage } from "@/components/sequence-models/HMMSequenceDetailPage";
import { MaskedTransformerJobsPage } from "@/components/sequence-models/MaskedTransformerJobsPage";
import { MaskedTransformerDetailPage } from "@/components/sequence-models/MaskedTransformerDetailPage";

export default function App() {
  return (
    <AppShell>
      <Routes>
        <Route path="/" element={<Navigate to="/app/call-parsing/detection" replace />} />
        <Route path="/app/classifier" element={<Navigate to="/app/classifier/training" replace />} />
        <Route path="/app/classifier/training" element={<TrainingTab />} />
        <Route path="/app/classifier/hydrophone" element={<HydrophoneTab />} />
        <Route path="/app/classifier/labeling" element={<LabelingTab />} />
        <Route path="/app/classifier/embeddings" element={<EmbeddingsPage />} />
        <Route path="/app/classifier/tuning" element={<TuningTab />} />
        <Route path="/app/classifier/timeline/:jobId" element={<ClassifierTimeline />} />
        <Route path="/app/vocalization" element={<Navigate to="/app/vocalization/training" replace />} />
        <Route path="/app/vocalization/training" element={<VocalizationTrainingTab />} />
        <Route path="/app/vocalization/labeling" element={<VocalizationLabelingTab />} />
        <Route path="/app/vocalization/training-data" element={<TrainingDataView />} />
        <Route path="/app/vocalization/clustering/:jobId" element={<VocalizationClusteringDetail />} />
        <Route path="/app/vocalization/clustering" element={<VocalizationClusteringPage />} />
        <Route path="/app/call-parsing" element={<Navigate to="/app/call-parsing/detection" replace />} />
        <Route path="/app/call-parsing/detection" element={<DetectionPage />} />
        <Route path="/app/call-parsing/region-timeline/:jobId" element={<RegionDetectionTimeline />} />
        <Route path="/app/call-parsing/segment" element={<SegmentPage />} />
        <Route path="/app/call-parsing/segment-training" element={<SegmentTrainingPage />} />
        <Route path="/app/call-parsing/classify" element={<ClassifyPage />} />
        <Route path="/app/call-parsing/classify-training" element={<ClassifyTrainingPage />} />
        <Route path="/app/call-parsing/window-classify" element={<WindowClassifyPage />} />
        <Route path="/app/sequence-models" element={<Navigate to="/app/sequence-models/continuous-embedding" replace />} />
        <Route path="/app/sequence-models/continuous-embedding/:jobId" element={<ContinuousEmbeddingDetailPage />} />
        <Route path="/app/sequence-models/continuous-embedding" element={<ContinuousEmbeddingJobsPage />} />
        <Route path="/app/sequence-models/hmm-sequence/:jobId" element={<HMMSequenceDetailPage />} />
        <Route path="/app/sequence-models/hmm-sequence" element={<HMMSequenceJobsPage />} />
        <Route path="/app/sequence-models/masked-transformer/:jobId" element={<MaskedTransformerDetailPage />} />
        <Route path="/app/sequence-models/masked-transformer" element={<MaskedTransformerJobsPage />} />
        <Route path="/app/admin" element={<AdminTab />} />
        <Route path="*" element={<Navigate to="/app/call-parsing/detection" replace />} />
      </Routes>
      <Toaster />
    </AppShell>
  );
}

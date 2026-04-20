import { Routes, Route, Navigate } from "react-router-dom";
import { Toaster } from "@/components/ui/toaster";
import { AppShell } from "@/components/layout/AppShell";
import { AudioTab } from "@/components/audio/AudioTab";
import { ProcessingTab } from "@/components/processing/ProcessingTab";
import { ClusteringTab } from "@/components/clustering/ClusteringTab";
import { AdminTab } from "@/components/admin/AdminTab";
import { TrainingTab } from "@/components/classifier/TrainingTab";
import { HydrophoneTab } from "@/components/classifier/HydrophoneTab";
import { LabelingTab } from "@/components/classifier/LabelingTab";
import { SearchTab } from "@/components/search/SearchTab";
import { LabelProcessingTab } from "@/components/label-processing/LabelProcessingTab";
import { ClassifierTimeline } from "@/components/timeline/ClassifierTimeline";
import { EmbeddingsPage } from "@/components/classifier/EmbeddingsPage";
import { TuningTab } from "@/components/classifier/TuningTab";
import { VocalizationTrainingTab } from "@/components/vocalization/VocalizationTrainingTab";
import { VocalizationLabelingTab } from "@/components/vocalization/VocalizationLabelingTab";
import { TrainingDataView } from "@/components/vocalization/TrainingDataView";
import { DetectionPage } from "@/components/call-parsing/DetectionPage";
import { SegmentPage } from "@/components/call-parsing/SegmentPage";
import { SegmentTrainingPage } from "@/components/call-parsing/SegmentTrainingPage";
import { ClassifyPage } from "@/components/call-parsing/ClassifyPage";
import { ClassifyTrainingPage } from "@/components/call-parsing/ClassifyTrainingPage";
import { RegionDetectionTimeline } from "@/components/call-parsing/RegionDetectionTimeline";

export default function App() {
  return (
    <AppShell>
      <Routes>
        <Route path="/" element={<Navigate to="/app/audio" replace />} />
        <Route path="/app/audio/:audioId" element={<AudioTab />} />
        <Route path="/app/audio" element={<AudioTab />} />
        <Route path="/app/processing" element={<ProcessingTab />} />
        <Route path="/app/clustering/:jobId" element={<ClusteringTab />} />
        <Route path="/app/clustering" element={<ClusteringTab />} />
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
        <Route path="/app/call-parsing" element={<Navigate to="/app/call-parsing/detection" replace />} />
        <Route path="/app/call-parsing/detection" element={<DetectionPage />} />
        <Route path="/app/call-parsing/region-timeline/:jobId" element={<RegionDetectionTimeline />} />
        <Route path="/app/call-parsing/segment" element={<SegmentPage />} />
        <Route path="/app/call-parsing/segment-training" element={<SegmentTrainingPage />} />
        <Route path="/app/call-parsing/classify" element={<ClassifyPage />} />
        <Route path="/app/call-parsing/classify-training" element={<ClassifyTrainingPage />} />
        <Route path="/app/search" element={<SearchTab />} />
        <Route path="/app/label-processing" element={<LabelProcessingTab />} />
        <Route path="/app/admin" element={<AdminTab />} />
        <Route path="*" element={<Navigate to="/app/audio" replace />} />
      </Routes>
      <Toaster />
    </AppShell>
  );
}

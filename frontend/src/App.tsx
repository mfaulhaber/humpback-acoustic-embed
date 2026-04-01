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
import { TimelineViewer } from "@/components/timeline/TimelineViewer";
import { VocalizationTrainingTab } from "@/components/vocalization/VocalizationTrainingTab";
import { VocalizationLabelingTab } from "@/components/vocalization/VocalizationLabelingTab";
import { TrainingDataView } from "@/components/vocalization/TrainingDataView";

export default function App() {
  return (
    <Routes>
      {/* Timeline viewer renders full-screen, outside the AppShell layout */}
      <Route path="/app/classifier/timeline/:jobId" element={<><TimelineViewer /><Toaster /></>} />
      <Route path="*" element={
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
            <Route path="/app/vocalization" element={<Navigate to="/app/vocalization/training" replace />} />
            <Route path="/app/vocalization/training" element={<VocalizationTrainingTab />} />
            <Route path="/app/vocalization/labeling" element={<VocalizationLabelingTab />} />
            <Route path="/app/vocalization/training-data" element={<TrainingDataView />} />
            <Route path="/app/search" element={<SearchTab />} />
            <Route path="/app/label-processing" element={<LabelProcessingTab />} />
            <Route path="/app/admin" element={<AdminTab />} />
            <Route path="*" element={<Navigate to="/app/audio" replace />} />
          </Routes>
          <Toaster />
        </AppShell>
      } />
    </Routes>
  );
}

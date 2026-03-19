import { Routes, Route, Navigate } from "react-router-dom";
import { Toaster } from "@/components/ui/toaster";
import { AppShell } from "@/components/layout/AppShell";
import { AudioTab } from "@/components/audio/AudioTab";
import { ProcessingTab } from "@/components/processing/ProcessingTab";
import { ClusteringTab } from "@/components/clustering/ClusteringTab";
import { AdminTab } from "@/components/admin/AdminTab";
import { TrainingTab } from "@/components/classifier/TrainingTab";
import { HydrophoneTab } from "@/components/classifier/HydrophoneTab";
import { SearchTab } from "@/components/search/SearchTab";

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
        <Route path="/app/search" element={<SearchTab />} />
        <Route path="/app/admin" element={<AdminTab />} />
        <Route path="*" element={<Navigate to="/app/audio" replace />} />
      </Routes>
      <Toaster />
    </AppShell>
  );
}

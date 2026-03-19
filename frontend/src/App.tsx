import { Routes, Route, Navigate, useLocation } from "react-router-dom";
import { Toaster } from "@/components/ui/toaster";
import { AppShell } from "@/components/layout/AppShell";
import { AudioTab } from "@/components/audio/AudioTab";
import { ProcessingTab } from "@/components/processing/ProcessingTab";
import { ClusteringTab } from "@/components/clustering/ClusteringTab";
import { AdminTab } from "@/components/admin/AdminTab";
import { ClassifierTab } from "@/components/classifier/ClassifierTab";
import { SearchTab } from "@/components/search/SearchTab";

export type TabId = "audio" | "processing" | "clustering" | "classifier" | "search" | "admin";

function useActiveTab(): TabId {
  const { pathname } = useLocation();
  if (pathname.startsWith("/app/processing")) return "processing";
  if (pathname.startsWith("/app/clustering")) return "clustering";
  if (pathname.startsWith("/app/classifier")) return "classifier";
  if (pathname.startsWith("/app/search")) return "search";
  if (pathname.startsWith("/app/admin")) return "admin";
  return "audio";
}

export default function App() {
  const activeTab = useActiveTab();

  return (
    <AppShell activeTab={activeTab}>
      <Routes>
        <Route path="/" element={<Navigate to="/app/audio" replace />} />
        <Route path="/app/audio/:audioId" element={<AudioTab />} />
        <Route path="/app/audio" element={<AudioTab />} />
        <Route path="/app/processing" element={<ProcessingTab />} />
        <Route path="/app/clustering/:jobId" element={<ClusteringTab />} />
        <Route path="/app/clustering" element={<ClusteringTab />} />
        <Route path="/app/classifier" element={<ClassifierTab />} />
        <Route path="/app/search" element={<SearchTab />} />
        <Route path="/app/admin" element={<AdminTab />} />
        <Route path="*" element={<Navigate to="/app/audio" replace />} />
      </Routes>
      <Toaster />
    </AppShell>
  );
}

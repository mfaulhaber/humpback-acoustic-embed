import { useState } from "react";
import { Toaster } from "@/components/ui/toaster";
import { AppShell } from "@/components/layout/AppShell";
import { AudioTab } from "@/components/audio/AudioTab";
import { ProcessingTab } from "@/components/processing/ProcessingTab";
import { ClusteringTab } from "@/components/clustering/ClusteringTab";
import { AdminTab } from "@/components/admin/AdminTab";

export type TabId = "audio" | "processing" | "clustering" | "admin";

export default function App() {
  const [activeTab, setActiveTab] = useState<TabId>("audio");

  return (
    <AppShell activeTab={activeTab} onTabChange={setActiveTab}>
      {activeTab === "audio" && <AudioTab />}
      {activeTab === "processing" && <ProcessingTab />}
      {activeTab === "clustering" && <ClusteringTab />}
      {activeTab === "admin" && <AdminTab />}
      <Toaster />
    </AppShell>
  );
}

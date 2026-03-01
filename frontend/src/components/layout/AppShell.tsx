import type { ReactNode } from "react";
import type { TabId } from "@/App";
import { Header } from "./Header";
import { TabNav } from "./TabNav";

interface AppShellProps {
  activeTab: TabId;
  onTabChange: (tab: TabId) => void;
  children: ReactNode;
}

export function AppShell({ activeTab, onTabChange, children }: AppShellProps) {
  return (
    <div className="min-h-screen flex flex-col">
      <Header />
      <TabNav activeTab={activeTab} onTabChange={onTabChange} />
      <main className="flex-1 p-4 max-w-[1400px] w-full mx-auto">{children}</main>
    </div>
  );
}

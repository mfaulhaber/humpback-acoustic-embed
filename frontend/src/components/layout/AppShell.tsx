import type { ReactNode } from "react";
import type { TabId } from "@/App";
import { Header } from "./Header";
import { TabNav } from "./TabNav";
import { DatabaseErrorBanner } from "@/components/shared/DatabaseErrorBanner";

interface AppShellProps {
  activeTab: TabId;
  children: ReactNode;
}

export function AppShell({ activeTab, children }: AppShellProps) {
  return (
    <div className="min-h-screen flex flex-col">
      <Header />
      <TabNav activeTab={activeTab} />
      <DatabaseErrorBanner />
      <main className="flex-1 p-4 max-w-[1400px] w-full mx-auto">{children}</main>
    </div>
  );
}

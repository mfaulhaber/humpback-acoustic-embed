import type { ReactNode } from "react";
import { TopNav } from "./TopNav";
import { SideNav } from "./SideNav";
import { Breadcrumbs } from "./Breadcrumbs";
import { DatabaseErrorBanner } from "@/components/shared/DatabaseErrorBanner";

interface AppShellProps {
  children: ReactNode;
}

export function AppShell({ children }: AppShellProps) {
  return (
    <div className="min-h-screen">
      <TopNav />
      <SideNav />
      <main className="ml-60 mt-12 p-4">
        <div className="max-w-[1400px] mx-auto">
          <Breadcrumbs />
          <DatabaseErrorBanner />
          {children}
        </div>
      </main>
    </div>
  );
}

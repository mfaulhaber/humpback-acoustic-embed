import { NavLink } from "react-router-dom";
import type { TabId } from "@/App";
import { cn } from "@/lib/utils";
import { Music, Cpu, Network, Settings } from "lucide-react";

const tabs: { id: TabId; label: string; icon: typeof Music; to: string }[] = [
  { id: "audio", label: "Audio", icon: Music, to: "/app/audio" },
  { id: "processing", label: "Processing", icon: Cpu, to: "/app/processing" },
  { id: "clustering", label: "Clustering", icon: Network, to: "/app/clustering" },
  { id: "admin", label: "Admin", icon: Settings, to: "/app/admin" },
];

interface TabNavProps {
  activeTab: TabId;
}

export function TabNav({ activeTab }: TabNavProps) {
  return (
    <nav className="border-b bg-white">
      <div className="max-w-[1400px] mx-auto flex">
        {tabs.map(({ id, label, icon: Icon, to }) => (
          <NavLink
            key={id}
            to={to}
            className={cn(
              "flex items-center gap-2 px-5 py-3 text-sm font-medium border-b-2 transition-colors",
              activeTab === id
                ? "border-slate-800 text-slate-800"
                : "border-transparent text-muted-foreground hover:text-foreground hover:border-slate-300"
            )}
          >
            <Icon className="h-4 w-4" />
            {label}
          </NavLink>
        ))}
      </div>
    </nav>
  );
}

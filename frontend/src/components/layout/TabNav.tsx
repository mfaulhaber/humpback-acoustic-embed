import type { TabId } from "@/App";
import { cn } from "@/lib/utils";
import { Music, Cpu, Network, Settings } from "lucide-react";

const tabs: { id: TabId; label: string; icon: typeof Music }[] = [
  { id: "audio", label: "Audio", icon: Music },
  { id: "processing", label: "Processing", icon: Cpu },
  { id: "clustering", label: "Clustering", icon: Network },
  { id: "admin", label: "Admin", icon: Settings },
];

interface TabNavProps {
  activeTab: TabId;
  onTabChange: (tab: TabId) => void;
}

export function TabNav({ activeTab, onTabChange }: TabNavProps) {
  return (
    <nav className="border-b bg-white">
      <div className="max-w-[1400px] mx-auto flex">
        {tabs.map(({ id, label, icon: Icon }) => (
          <button
            key={id}
            onClick={() => onTabChange(id)}
            className={cn(
              "flex items-center gap-2 px-5 py-3 text-sm font-medium border-b-2 transition-colors",
              activeTab === id
                ? "border-slate-800 text-slate-800"
                : "border-transparent text-muted-foreground hover:text-foreground hover:border-slate-300"
            )}
          >
            <Icon className="h-4 w-4" />
            {label}
          </button>
        ))}
      </div>
    </nav>
  );
}

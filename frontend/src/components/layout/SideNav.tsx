import { NavLink, useLocation } from "react-router-dom";
import { cn } from "@/lib/utils";
import { Music, Cpu, Network, Zap, AudioWaveform, Search, Settings, ChevronRight, FileAudio, Activity } from "lucide-react";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";

interface NavLeaf {
  label: string;
  icon: typeof Music;
  to: string;
}

interface NavGroup {
  label: string;
  icon: typeof Music;
  children: { label: string; to: string }[];
}

type NavItem = NavLeaf | NavGroup;

function isGroup(item: NavItem): item is NavGroup {
  return "children" in item;
}

const navItems: NavItem[] = [
  { label: "Audio", icon: Music, to: "/app/audio" },
  { label: "Processing", icon: Cpu, to: "/app/processing" },
  { label: "Clustering", icon: Network, to: "/app/clustering" },
  {
    label: "Classifier",
    icon: Zap,
    children: [
      { label: "Training", to: "/app/classifier/training" },
      { label: "Hydrophone Detection", to: "/app/classifier/hydrophone" },
      { label: "Embeddings", to: "/app/classifier/embeddings" },
      { label: "Tuning", to: "/app/classifier/tuning" },
    ],
  },
  {
    label: "Vocalization",
    icon: AudioWaveform,
    children: [
      { label: "Training", to: "/app/vocalization/training" },
      { label: "Labeling", to: "/app/vocalization/labeling" },
      { label: "Training Data", to: "/app/vocalization/training-data" },
    ],
  },
  {
    label: "Call Parsing",
    icon: Activity,
    children: [
      { label: "Detection", to: "/app/call-parsing/detection" },
      { label: "Segment", to: "/app/call-parsing/segment" },
      { label: "Segment Training", to: "/app/call-parsing/segment-training" },
      { label: "Classify", to: "/app/call-parsing/classify" },
      { label: "Classify Training", to: "/app/call-parsing/classify-training" },
    ],
  },
  { label: "Search", icon: Search, to: "/app/search" },
  { label: "Label Processing", icon: FileAudio, to: "/app/label-processing" },
  { label: "Admin", icon: Settings, to: "/app/admin" },
];

const linkClass =
  "flex items-center gap-3 px-4 py-2 text-sm transition-colors rounded-md mx-2";
const activeClass = "bg-slate-100 text-slate-900 font-medium";
const inactiveClass = "text-slate-600 hover:bg-slate-50 hover:text-slate-900";

function LeafLink({ item }: { item: NavLeaf }) {
  return (
    <NavLink
      to={item.to}
      end={item.to === "/app/audio"}
      className={({ isActive }) =>
        cn(linkClass, isActive ? activeClass : inactiveClass)
      }
    >
      <item.icon className="h-4 w-4 shrink-0" />
      {item.label}
    </NavLink>
  );
}

function GroupNav({ item }: { item: NavGroup }) {
  const { pathname } = useLocation();
  const isChildActive = item.children.some((c) =>
    pathname.startsWith(c.to),
  );

  return (
    <Collapsible defaultOpen={isChildActive}>
      <CollapsibleTrigger className={cn(linkClass, "w-[calc(100%-1rem)] justify-between group", inactiveClass)}>
        <span className="flex items-center gap-3">
          <item.icon className="h-4 w-4 shrink-0" />
          {item.label}
        </span>
        <ChevronRight className="h-3.5 w-3.5 transition-transform group-data-[state=open]:rotate-90" />
      </CollapsibleTrigger>
      <CollapsibleContent>
        <div className="ml-6 mt-0.5 space-y-0.5">
          {item.children.map((child) => (
            <NavLink
              key={child.to}
              to={child.to}
              className={({ isActive }) =>
                cn(
                  "flex items-center px-4 py-1.5 text-sm rounded-md mx-2 transition-colors",
                  isActive ? activeClass : inactiveClass,
                )
              }
            >
              {child.label}
            </NavLink>
          ))}
        </div>
      </CollapsibleContent>
    </Collapsible>
  );
}

export function SideNav() {
  return (
    <nav className="fixed top-12 left-0 bottom-0 w-60 border-r bg-white overflow-y-auto z-40">
      <div className="py-3 space-y-0.5">
        {navItems.map((item) =>
          isGroup(item) ? (
            <GroupNav key={item.label} item={item} />
          ) : (
            <LeafLink key={item.label} item={item} />
          ),
        )}
      </div>
    </nav>
  );
}

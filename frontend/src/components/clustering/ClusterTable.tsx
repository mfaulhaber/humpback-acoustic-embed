import { useState, useCallback, useMemo } from "react";
import { ChevronRight } from "lucide-react";
import { useAssignments } from "@/hooks/queries/useClustering";
import { useAudioFiles } from "@/hooks/queries/useAudioFiles";
import { useEmbeddingSets } from "@/hooks/queries/useProcessing";
import { AudioPlayerBar } from "@/components/audio/AudioPlayerBar";
import { FolderTree } from "@/components/shared/FolderTree";
import { cn } from "@/lib/utils";
import type { ClusterOut } from "@/api/types";

interface ClusterTableProps {
  clusters: ClusterOut[];
}

export function ClusterTable({ clusters }: ClusterTableProps) {
  const [expandedId, setExpandedId] = useState<string | null>(null);
  const sorted = useMemo(() => [...clusters].sort((a, b) => a.cluster_label - b.cluster_label), [clusters]);

  return (
    <div className="border rounded-md">
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b bg-muted/50">
            <th className="text-left py-2 px-3 font-medium w-8"></th>
            <th className="text-left py-2 px-3 font-medium">Label</th>
            <th className="text-left py-2 px-3 font-medium">Size</th>
            <th className="text-left py-2 px-3 font-medium">Metadata</th>
          </tr>
        </thead>
        <tbody>
          {sorted.map((cluster) => (
            <ClusterRow
              key={cluster.id}
              cluster={cluster}
              isExpanded={expandedId === cluster.id}
              onToggle={() => setExpandedId(expandedId === cluster.id ? null : cluster.id)}
            />
          ))}
        </tbody>
      </table>
    </div>
  );
}

interface ClusterRowProps {
  cluster: ClusterOut;
  isExpanded: boolean;
  onToggle: () => void;
}

function ClusterRow({ cluster, isExpanded, onToggle }: ClusterRowProps) {
  return (
    <>
      <tr
        className="border-b cursor-pointer hover:bg-accent"
        onClick={onToggle}
      >
        <td className="py-2 px-3">
          <ChevronRight
            className={cn("h-4 w-4 transition-transform", isExpanded && "rotate-90")}
          />
        </td>
        <td className="py-2 px-3 font-medium">
          {cluster.cluster_label === -1 ? "Noise" : `Cluster ${cluster.cluster_label}`}
        </td>
        <td className="py-2 px-3">{cluster.size}</td>
        <td className="py-2 px-3 text-xs text-muted-foreground">
          {cluster.metadata_summary
            ? Object.entries(cluster.metadata_summary)
                .map(([k, v]) => `${k}: ${v}`)
                .join(", ")
            : "—"}
        </td>
      </tr>
      {isExpanded && (
        <tr>
          <td colSpan={4} className="p-3 bg-muted/30">
            <ClusterAssignments clusterId={cluster.id} />
          </td>
        </tr>
      )}
    </>
  );
}

function ClusterAssignments({ clusterId }: { clusterId: string }) {
  const { data: assignments = [], isLoading } = useAssignments(clusterId);
  const { data: audioFiles = [] } = useAudioFiles();
  const { data: embeddingSets = [] } = useEmbeddingSets();

  const audioMap = new Map(audioFiles.map((af) => [af.id, af]));
  const esMap = new Map(embeddingSets.map((es) => [es.id, es]));

  // Group assignments by audio file
  const grouped = useMemo(() => {
    const map = new Map<
      string,
      { audioId: string; filename: string; folderPath: string; windowSize: number; duration: number; windows: number[] }
    >();

    for (const a of assignments) {
      const es = esMap.get(a.embedding_set_id);
      const audioId = es?.audio_file_id ?? "";
      const af = audioMap.get(audioId);
      if (!map.has(audioId)) {
        map.set(audioId, {
          audioId,
          filename: af?.filename ?? a.embedding_set_id,
          folderPath: af?.folder_path ?? "",
          windowSize: es?.window_size_seconds ?? 5,
          duration: af?.duration_seconds ?? 0,
          windows: [],
        });
      }
      map.get(audioId)!.windows.push(a.embedding_row_index);
    }

    return Array.from(map.values());
  }, [assignments, audioMap, esMap]);

  if (isLoading) return <p className="text-sm text-muted-foreground">Loading assignments...</p>;
  if (assignments.length === 0) return <p className="text-sm text-muted-foreground">No assignments.</p>;

  return (
    <FolderTree
      items={grouped}
      getPath={(g) => g.folderPath}
      stateKey={`clusterAssign-${clusterId}`}
      renderLeaf={(g) => (
        <div className="py-2">
          <p className="text-sm font-medium mb-1">{g.filename}</p>
          <AudioPlayerBar
            audioId={g.audioId}
            totalWindows={Math.max(...g.windows) + 1}
            windowSizeSeconds={g.windowSize}
            duration={g.duration}
            activeWindow={g.windows[0] ?? 0}
            onWindowClick={() => {}}
            maxChips={40}
          />
        </div>
      )}
    />
  );
}

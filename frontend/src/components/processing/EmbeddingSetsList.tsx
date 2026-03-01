import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { FolderTree } from "@/components/shared/FolderTree";
import { useAudioFiles } from "@/hooks/queries/useAudioFiles";
import { shortId, fmtDate } from "@/utils/format";
import type { EmbeddingSet } from "@/api/types";

interface EmbeddingSetsListProps {
  embeddingSets: EmbeddingSet[];
}

export function EmbeddingSetsList({ embeddingSets }: EmbeddingSetsListProps) {
  const { data: audioFiles = [] } = useAudioFiles();
  const audioMap = new Map(audioFiles.map((af) => [af.id, af]));

  type EnrichedES = EmbeddingSet & { _folderPath: string; _filename: string };
  const enriched: EnrichedES[] = embeddingSets.map((es) => {
    const af = audioMap.get(es.audio_file_id);
    return {
      ...es,
      _folderPath: af?.folder_path ?? "",
      _filename: af?.filename ?? es.audio_file_id,
    };
  });

  return (
    <Card>
      <CardHeader className="pb-3">
        <CardTitle className="text-base">Embedding Sets ({embeddingSets.length})</CardTitle>
      </CardHeader>
      <CardContent>
        {embeddingSets.length === 0 ? (
          <p className="text-sm text-muted-foreground">No embedding sets yet.</p>
        ) : (
          <FolderTree
            items={enriched}
            getPath={(es) => es._folderPath}
            stateKey="embedTree"
            renderLeaf={(es) => (
              <div className="flex items-center gap-2 py-1 px-2 text-sm hover:bg-accent rounded">
                <span className="font-mono text-xs text-muted-foreground">{shortId(es.id)}</span>
                <span className="truncate">{es._filename}</span>
                <span className="text-xs text-muted-foreground">{es.model_version}</span>
                <span className="text-xs text-muted-foreground">{es.vector_dim}d</span>
                <span className="text-xs text-muted-foreground ml-auto">{fmtDate(es.created_at)}</span>
              </div>
            )}
          />
        )}
      </CardContent>
    </Card>
  );
}

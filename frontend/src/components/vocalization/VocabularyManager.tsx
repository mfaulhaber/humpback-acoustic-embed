import { useState, useMemo } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { Checkbox } from "@/components/ui/checkbox";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import { Plus, Pencil, Trash2, Download, Loader2 } from "lucide-react";
import {
  useVocalizationTypes,
  useCreateVocalizationType,
  useUpdateVocalizationType,
  useDeleteVocalizationType,
  useImportVocalizationTypes,
} from "@/hooks/queries/useVocalization";
import { useEmbeddingSets } from "@/hooks/queries/useProcessing";
import { useAudioFiles } from "@/hooks/queries/useAudioFiles";
import { fmtDate } from "@/utils/format";

export function VocabularyManager() {
  const { data: types = [], isLoading } = useVocalizationTypes();
  const createMut = useCreateVocalizationType();
  const updateMut = useUpdateVocalizationType();
  const deleteMut = useDeleteVocalizationType();

  const [addName, setAddName] = useState("");
  const [addDesc, setAddDesc] = useState("");
  const [editId, setEditId] = useState<string | null>(null);
  const [editName, setEditName] = useState("");
  const [editDesc, setEditDesc] = useState("");
  const [deleteConfirmId, setDeleteConfirmId] = useState<string | null>(null);

  function handleAdd() {
    if (!addName.trim()) return;
    createMut.mutate(
      { name: addName.trim(), description: addDesc.trim() || undefined },
      {
        onSuccess: () => {
          setAddName("");
          setAddDesc("");
        },
      },
    );
  }

  function startEdit(id: string, name: string, desc: string | null) {
    setEditId(id);
    setEditName(name);
    setEditDesc(desc ?? "");
  }

  function handleEdit() {
    if (!editId) return;
    updateMut.mutate(
      {
        typeId: editId,
        body: { name: editName.trim() || undefined, description: editDesc.trim() || undefined },
      },
      { onSuccess: () => setEditId(null) },
    );
  }

  function handleDelete() {
    if (!deleteConfirmId) return;
    deleteMut.mutate(deleteConfirmId, {
      onSuccess: () => setDeleteConfirmId(null),
    });
  }

  return (
    <Card>
      <CardHeader className="flex flex-row items-center justify-between pb-3">
        <CardTitle className="text-base">Vocabulary</CardTitle>
        <ImportDialog />
      </CardHeader>
      <CardContent className="space-y-3">
        {/* Add form */}
        <div className="flex gap-2">
          <Input
            placeholder="Type name"
            value={addName}
            onChange={(e) => setAddName(e.target.value)}
            className="h-8 text-sm"
            onKeyDown={(e) => e.key === "Enter" && handleAdd()}
          />
          <Input
            placeholder="Description (optional)"
            value={addDesc}
            onChange={(e) => setAddDesc(e.target.value)}
            className="h-8 text-sm"
            onKeyDown={(e) => e.key === "Enter" && handleAdd()}
          />
          <Button
            size="sm"
            onClick={handleAdd}
            disabled={!addName.trim() || createMut.isPending}
          >
            <Plus className="h-3.5 w-3.5 mr-1" />
            Add
          </Button>
        </div>

        {/* Type list */}
        {isLoading ? (
          <p className="text-sm text-muted-foreground">Loading...</p>
        ) : types.length === 0 ? (
          <p className="text-sm text-muted-foreground">
            No vocalization types defined. Add one above or import from embedding sets.
          </p>
        ) : (
          <div className="border rounded-md divide-y">
            {types.map((t) => (
              <div key={t.id} className="flex items-center justify-between px-3 py-2 text-sm">
                {editId === t.id ? (
                  <div className="flex gap-2 flex-1 mr-2">
                    <Input
                      value={editName}
                      onChange={(e) => setEditName(e.target.value)}
                      className="h-7 text-sm"
                    />
                    <Input
                      value={editDesc}
                      onChange={(e) => setEditDesc(e.target.value)}
                      className="h-7 text-sm"
                      placeholder="Description"
                    />
                    <Button size="sm" variant="outline" className="h-7" onClick={handleEdit}>
                      Save
                    </Button>
                    <Button
                      size="sm"
                      variant="ghost"
                      className="h-7"
                      onClick={() => setEditId(null)}
                    >
                      Cancel
                    </Button>
                  </div>
                ) : (
                  <>
                    <div className="flex items-center gap-2 min-w-0">
                      <Badge variant="secondary">{t.name}</Badge>
                      {t.description && (
                        <span className="text-muted-foreground truncate">{t.description}</span>
                      )}
                      <span className="text-xs text-muted-foreground whitespace-nowrap">
                        {fmtDate(t.created_at)}
                      </span>
                    </div>
                    <div className="flex items-center gap-1 shrink-0">
                      <Button
                        size="icon"
                        variant="ghost"
                        className="h-7 w-7"
                        onClick={() => startEdit(t.id, t.name, t.description)}
                      >
                        <Pencil className="h-3.5 w-3.5" />
                      </Button>
                      <Button
                        size="icon"
                        variant="ghost"
                        className="h-7 w-7 text-destructive"
                        onClick={() => setDeleteConfirmId(t.id)}
                      >
                        <Trash2 className="h-3.5 w-3.5" />
                      </Button>
                    </div>
                  </>
                )}
              </div>
            ))}
          </div>
        )}

        {/* Delete confirmation dialog */}
        <Dialog
          open={deleteConfirmId !== null}
          onOpenChange={(open) => !open && setDeleteConfirmId(null)}
        >
          <DialogContent>
            <DialogHeader>
              <DialogTitle>Delete vocalization type?</DialogTitle>
              <DialogDescription>
                This cannot be undone. If this type is part of an active model&apos;s vocabulary,
                deletion will fail.
              </DialogDescription>
            </DialogHeader>
            <DialogFooter>
              <Button variant="outline" onClick={() => setDeleteConfirmId(null)}>
                Cancel
              </Button>
              <Button
                variant="destructive"
                onClick={handleDelete}
                disabled={deleteMut.isPending}
              >
                Delete
              </Button>
            </DialogFooter>
            {deleteMut.isError && (
              <p className="text-sm text-destructive px-6 pb-4">
                {(deleteMut.error as Error).message}
              </p>
            )}
          </DialogContent>
        </Dialog>
      </CardContent>
    </Card>
  );
}

function ImportDialog() {
  const { data: embeddingSets = [] } = useEmbeddingSets();
  const { data: audioFiles = [] } = useAudioFiles();
  const importMut = useImportVocalizationTypes();
  const [open, setOpen] = useState(false);
  const [selected, setSelected] = useState<Set<string>>(new Set());

  // Group embedding sets by parent folder (dataset name)
  const datasetGroups = useMemo(() => {
    const audioMap = new Map(audioFiles.map((af) => [af.id, af.folder_path]));
    const groups = new Map<string, string[]>();
    for (const es of embeddingSets) {
      const folderPath = audioMap.get(es.audio_file_id) ?? "";
      // Parent folder = first path segment (the dataset name)
      const slashIdx = folderPath.indexOf("/");
      const parent = slashIdx >= 0 ? folderPath.slice(0, slashIdx) : folderPath;
      if (!parent) continue;
      const ids = groups.get(parent);
      if (ids) ids.push(es.id);
      else groups.set(parent, [es.id]);
    }
    return Array.from(groups.entries())
      .map(([dataset, ids]) => ({ dataset, ids }))
      .sort((a, b) => a.dataset.localeCompare(b.dataset));
  }, [embeddingSets, audioFiles]);

  function toggle(dataset: string) {
    setSelected((prev) => {
      const next = new Set(prev);
      if (next.has(dataset)) next.delete(dataset);
      else next.add(dataset);
      return next;
    });
  }

  // Expand selected datasets to their embedding set IDs
  function selectedEmbSetIds(): string[] {
    const ids: string[] = [];
    for (const g of datasetGroups) {
      if (selected.has(g.dataset)) ids.push(...g.ids);
    }
    return ids;
  }

  function handleImport() {
    importMut.mutate(
      { embedding_set_ids: selectedEmbSetIds() },
      {
        onSuccess: () => {
          setOpen(false);
          setSelected(new Set());
        },
      },
    );
  }

  return (
    <Dialog open={open} onOpenChange={setOpen}>
      <DialogTrigger asChild>
        <Button size="sm" variant="outline">
          <Download className="h-3.5 w-3.5 mr-1" />
          Import
        </Button>
      </DialogTrigger>
      <DialogContent className="max-w-lg max-h-[70vh] flex flex-col">
        <DialogHeader>
          <DialogTitle>Import types from embedding sets</DialogTitle>
          <DialogDescription>
            Select datasets to scan. Subfolder names within each dataset will be imported as
            vocalization types.
          </DialogDescription>
        </DialogHeader>
        <div className="flex-1 overflow-y-auto border rounded-md divide-y">
          {datasetGroups.length === 0 ? (
            <p className="p-3 text-sm text-muted-foreground">No embedding set datasets available.</p>
          ) : (
            datasetGroups.map((g) => (
              <label
                key={g.dataset}
                className="flex items-center gap-2 px-3 py-2 text-sm cursor-pointer hover:bg-muted/50"
              >
                <Checkbox
                  checked={selected.has(g.dataset)}
                  onCheckedChange={() => toggle(g.dataset)}
                />
                <span className="truncate font-medium">{g.dataset}</span>
                <span className="text-xs text-muted-foreground shrink-0">
                  ({g.ids.length} {g.ids.length === 1 ? "file" : "files"})
                </span>
              </label>
            ))
          )}
        </div>
        {importMut.isSuccess && importMut.data && (
          <div className="text-sm space-y-1">
            <p className="text-green-600">
              Added: {importMut.data.added.length > 0 ? importMut.data.added.join(", ") : "none"}
            </p>
            <p className="text-muted-foreground">
              Skipped (already exist):{" "}
              {importMut.data.skipped.length > 0 ? importMut.data.skipped.join(", ") : "none"}
            </p>
          </div>
        )}
        <DialogFooter>
          <Button variant="outline" onClick={() => setOpen(false)}>
            Cancel
          </Button>
          <Button
            onClick={handleImport}
            disabled={selected.size === 0 || importMut.isPending}
          >
            {importMut.isPending && <Loader2 className="h-3.5 w-3.5 mr-1 animate-spin" />}
            Import ({selected.size} {selected.size === 1 ? "dataset" : "datasets"})
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}

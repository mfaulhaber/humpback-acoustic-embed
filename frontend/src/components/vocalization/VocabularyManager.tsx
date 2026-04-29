import { useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Plus, Pencil, Trash2 } from "lucide-react";
import {
  useVocalizationTypes,
  useCreateVocalizationType,
  useUpdateVocalizationType,
  useDeleteVocalizationType,
} from "@/hooks/queries/useVocalization";
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
        body: {
          name: editName.trim() || undefined,
          description: editDesc.trim() || undefined,
        },
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
      <CardHeader className="pb-3">
        <CardTitle className="text-base">Vocabulary</CardTitle>
      </CardHeader>
      <CardContent className="space-y-3">
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

        {isLoading ? (
          <p className="text-sm text-muted-foreground">Loading...</p>
        ) : types.length === 0 ? (
          <p className="text-sm text-muted-foreground">
            No vocalization types defined. Add one above to start labeling.
          </p>
        ) : (
          <div className="border rounded-md divide-y">
            {types.map((t) => (
              <div
                key={t.id}
                className="flex items-center justify-between px-3 py-2 text-sm"
              >
                {editId === t.id ? (
                  <div className="flex flex-1 gap-2 mr-2">
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

        <Dialog
          open={deleteConfirmId !== null}
          onOpenChange={(open) => !open && setDeleteConfirmId(null)}
        >
          <DialogContent>
            <DialogHeader>
              <DialogTitle>Delete vocalization type?</DialogTitle>
              <DialogDescription>
                This cannot be undone. If this type is part of an active model&apos;s
                vocabulary, deletion will fail.
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

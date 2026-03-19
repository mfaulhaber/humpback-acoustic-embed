import { ChevronRight, ChevronDown } from "lucide-react";
import { Checkbox } from "@/components/ui/checkbox";
import { Badge } from "@/components/ui/badge";
import type { EmbeddingSet } from "@/api/types";

// ---- Types ----

export const ROOT_SENTINEL = "__root__";

export interface FolderNode {
  child: string;
  sets: EmbeddingSet[];
}

export interface ParentNode {
  parent: string;
  children: FolderNode[];
  totalSets: number;
}

// ---- Tree builder ----

export function buildFolderTree(
  embeddingSets: EmbeddingSet[],
  audioMap: Map<string, { folder_path: string; filename: string }>,
): ParentNode[] {
  const tree = new Map<string, Map<string, EmbeddingSet[]>>();

  for (const es of embeddingSets) {
    const af = audioMap.get(es.audio_file_id);
    const folderPath = af?.folder_path || "";
    const slashIdx = folderPath.indexOf("/");
    const parent = folderPath
      ? slashIdx >= 0
        ? folderPath.slice(0, slashIdx)
        : folderPath
      : ROOT_SENTINEL;
    const child =
      folderPath && slashIdx >= 0
        ? folderPath.slice(slashIdx + 1)
        : ROOT_SENTINEL;

    if (!tree.has(parent)) tree.set(parent, new Map());
    const childMap = tree.get(parent)!;
    if (!childMap.has(child)) childMap.set(child, []);
    childMap.get(child)!.push(es);
  }

  const result: ParentNode[] = [];
  const sortedParents = [...tree.keys()].sort((a, b) => {
    if (a === ROOT_SENTINEL) return -1;
    if (b === ROOT_SENTINEL) return 1;
    return a.localeCompare(b);
  });

  for (const parent of sortedParents) {
    const childMap = tree.get(parent)!;
    const sortedChildren = [...childMap.keys()].sort((a, b) => {
      if (a === ROOT_SENTINEL) return -1;
      if (b === ROOT_SENTINEL) return 1;
      return a.localeCompare(b);
    });
    const children: FolderNode[] = sortedChildren.map((child) => ({
      child,
      sets: childMap.get(child)!,
    }));
    const totalSets = children.reduce((sum, c) => sum + c.sets.length, 0);
    result.push({ parent, children, totalSets });
  }

  return result;
}

// ---- Toggle helper factories ----

export function makeToggleChild(
  setSel: React.Dispatch<React.SetStateAction<Set<string>>>,
) {
  return (sets: EmbeddingSet[]) => {
    setSel((prev) => {
      const next = new Set(prev);
      const allIn = sets.every((es) => next.has(es.id));
      for (const es of sets) {
        if (allIn) next.delete(es.id);
        else next.add(es.id);
      }
      return next;
    });
  };
}

export function makeToggleParent(
  setSel: React.Dispatch<React.SetStateAction<Set<string>>>,
) {
  return (node: ParentNode) => {
    setSel((prev) => {
      const next = new Set(prev);
      const allSets = node.children.flatMap((c) => c.sets);
      const allIn = allSets.every((es) => next.has(es.id));
      for (const es of allSets) {
        if (allIn) next.delete(es.id);
        else next.add(es.id);
      }
      return next;
    });
  };
}

export function makeToggleAll(
  filteredSets: EmbeddingSet[],
  sel: Set<string>,
  setSel: React.Dispatch<React.SetStateAction<Set<string>>>,
) {
  return () => {
    const allSel =
      filteredSets.length > 0 &&
      filteredSets.every((es) => sel.has(es.id));
    if (allSel) setSel(new Set());
    else setSel(new Set(filteredSets.map((es) => es.id)));
  };
}

export function makeToggleCollapse(
  allParentKeys: Set<string>,
  collapsed: Set<string> | null,
  setCollapsed: React.Dispatch<React.SetStateAction<Set<string> | null>>,
) {
  return (parent: string) => {
    setCollapsed((prev) => {
      const base = prev ?? allParentKeys;
      const next = new Set(base);
      if (next.has(parent)) next.delete(parent);
      else next.add(parent);
      return next;
    });
  };
}

// ---- Multi-select panel (folder-level leaves) ----

export function EmbeddingSetPanel({
  label,
  selected,
  collapsed,
  folderTree,
  embeddingSets,
  onToggleChild,
  onToggleParent,
  onToggleAll,
  onToggleCollapse,
  displayName,
}: {
  label: string;
  selected: Set<string>;
  collapsed: Set<string>;
  folderTree: ParentNode[];
  embeddingSets: EmbeddingSet[];
  onToggleChild: (sets: EmbeddingSet[]) => void;
  onToggleParent: (node: ParentNode) => void;
  onToggleAll: () => void;
  onToggleCollapse: (parent: string) => void;
  displayName: (key: string) => string;
}) {
  const allSelected =
    embeddingSets.length > 0 &&
    embeddingSets.every((es) => selected.has(es.id));
  const someSelected = embeddingSets.some((es) => selected.has(es.id));

  return (
    <div>
      <div className="space-y-1 max-h-72 overflow-y-auto border rounded p-2">
        <div className="flex items-center gap-2 pb-1 border-b mb-1">
          <Checkbox
            checked={
              allSelected ? true : someSelected ? "indeterminate" : false
            }
            onCheckedChange={onToggleAll}
          />
          <span className="text-sm font-medium">{label}</span>
        </div>
        {folderTree.map((node) => {
          const allParentSets = node.children.flatMap((c) => c.sets);
          const parentAllSelected = allParentSets.every((es) =>
            selected.has(es.id),
          );
          const parentSomeSelected = allParentSets.some((es) =>
            selected.has(es.id),
          );
          const isCollapsed = collapsed.has(node.parent);

          return (
            <div key={node.parent}>
              <div className="flex items-center gap-1.5 py-1">
                <button
                  className="p-0.5 hover:bg-muted rounded"
                  onClick={() => onToggleCollapse(node.parent)}
                >
                  {isCollapsed ? (
                    <ChevronRight className="h-3.5 w-3.5 text-muted-foreground" />
                  ) : (
                    <ChevronDown className="h-3.5 w-3.5 text-muted-foreground" />
                  )}
                </button>
                <Checkbox
                  checked={
                    parentAllSelected
                      ? true
                      : parentSomeSelected
                        ? "indeterminate"
                        : false
                  }
                  onCheckedChange={() => onToggleParent(node)}
                />
                <span
                  className="text-sm font-medium cursor-pointer select-none"
                  onClick={() => onToggleCollapse(node.parent)}
                >
                  {displayName(node.parent)}
                </span>
                <span className="text-xs text-muted-foreground">
                  ({node.totalSets} sets)
                </span>
              </div>

              {!isCollapsed && (
                <div className="ml-6 space-y-0.5">
                  {node.children.map((child) => {
                    const childAllSelected = child.sets.every((es) =>
                      selected.has(es.id),
                    );
                    const childSomeSelected = child.sets.some((es) =>
                      selected.has(es.id),
                    );
                    return (
                      <label
                        key={child.child}
                        className="flex items-center gap-2 py-0.5 text-sm cursor-pointer"
                      >
                        <Checkbox
                          checked={
                            childAllSelected
                              ? true
                              : childSomeSelected
                                ? "indeterminate"
                                : false
                          }
                          onCheckedChange={() => onToggleChild(child.sets)}
                        />
                        <span className="truncate">
                          {displayName(child.child)}
                        </span>
                        <Badge variant="outline" className="text-[10px] px-1.5 py-0 font-normal shrink-0">
                          {child.sets[0]?.model_version}
                        </Badge>
                        <span className="text-xs text-muted-foreground ml-auto shrink-0">
                          {child.sets.length}{" "}
                          {child.sets.length === 1 ? "set" : "sets"}
                        </span>
                      </label>
                    );
                  })}
                </div>
              )}
            </div>
          );
        })}
        {embeddingSets.length === 0 && (
          <p className="text-sm text-muted-foreground">
            No embedding sets available. Process audio first.
          </p>
        )}
      </div>
      {selected.size > 0 && (
        <p className="text-xs text-muted-foreground mt-1">
          {selected.size} of {embeddingSets.length} embedding sets selected
        </p>
      )}
    </div>
  );
}

// ---- Single-select panel (file-level leaves) ----

export function EmbeddingQueryPanel({
  label,
  selectedEsId,
  onSelectEs,
  collapsed,
  folderTree,
  embeddingSets,
  onToggleCollapse,
  displayName,
  audioMap,
}: {
  label: string;
  selectedEsId: string;
  onSelectEs: (esId: string) => void;
  collapsed: Set<string>;
  folderTree: ParentNode[];
  embeddingSets: EmbeddingSet[];
  onToggleCollapse: (parent: string) => void;
  displayName: (key: string) => string;
  audioMap: Map<string, { folder_path: string; filename: string }>;
}) {
  return (
    <div>
      <div className="space-y-1 max-h-72 overflow-y-auto border rounded p-2">
        <div className="flex items-center gap-2 pb-1 border-b mb-1">
          <span className="text-sm font-medium">{label}</span>
        </div>
        {folderTree.map((node) => {
          const isCollapsed = collapsed.has(node.parent);

          return (
            <div key={node.parent}>
              <div className="flex items-center gap-1.5 py-1">
                <button
                  className="p-0.5 hover:bg-muted rounded"
                  onClick={() => onToggleCollapse(node.parent)}
                >
                  {isCollapsed ? (
                    <ChevronRight className="h-3.5 w-3.5 text-muted-foreground" />
                  ) : (
                    <ChevronDown className="h-3.5 w-3.5 text-muted-foreground" />
                  )}
                </button>
                <span
                  className="text-sm font-medium cursor-pointer select-none"
                  onClick={() => onToggleCollapse(node.parent)}
                >
                  {displayName(node.parent)}
                </span>
                <span className="text-xs text-muted-foreground">
                  ({node.totalSets} files)
                </span>
              </div>

              {!isCollapsed && (
                <div className="ml-6 space-y-0.5">
                  {node.children.map((child) => {
                    const childKey = `${node.parent}/${child.child}`;
                    const isChildCollapsed = collapsed.has(childKey);

                    return (
                      <div key={child.child}>
                        <div className="flex items-center gap-1.5 py-0.5">
                          <button
                            className="p-0.5 hover:bg-muted rounded"
                            onClick={() => onToggleCollapse(childKey)}
                          >
                            {isChildCollapsed ? (
                              <ChevronRight className="h-3 w-3 text-muted-foreground" />
                            ) : (
                              <ChevronDown className="h-3 w-3 text-muted-foreground" />
                            )}
                          </button>
                          <span
                            className="text-xs font-medium text-muted-foreground cursor-pointer select-none"
                            onClick={() => onToggleCollapse(childKey)}
                          >
                            {displayName(child.child)}
                          </span>
                          <span className="text-xs text-muted-foreground">
                            ({child.sets.length} {child.sets.length === 1 ? "file" : "files"})
                          </span>
                        </div>
                        {!isChildCollapsed &&
                          child.sets.map((es) => {
                            const af = audioMap.get(es.audio_file_id);
                            const isSelected = selectedEsId === es.id;
                            return (
                              <label
                                key={es.id}
                                className={`flex items-center gap-2 py-0.5 text-sm cursor-pointer rounded px-1 ml-5 ${isSelected ? "bg-accent" : "hover:bg-muted/50"}`}
                                onClick={() => onSelectEs(es.id)}
                              >
                                <span
                                  className={`h-3.5 w-3.5 rounded-full border-2 shrink-0 flex items-center justify-center ${isSelected ? "border-primary" : "border-muted-foreground/40"}`}
                                >
                                  {isSelected && (
                                    <span className="h-1.5 w-1.5 rounded-full bg-primary" />
                                  )}
                                </span>
                                <span className="truncate">
                                  {af?.filename ?? es.id.slice(0, 8)}
                                </span>
                                <Badge variant="outline" className="text-[10px] px-1.5 py-0 font-normal shrink-0">
                                  {es.model_version}
                                </Badge>
                              </label>
                            );
                          })}
                      </div>
                    );
                  })}
                </div>
              )}
            </div>
          );
        })}
        {embeddingSets.length === 0 && (
          <p className="text-sm text-muted-foreground">
            No embedding sets available. Process audio first.
          </p>
        )}
      </div>
    </div>
  );
}

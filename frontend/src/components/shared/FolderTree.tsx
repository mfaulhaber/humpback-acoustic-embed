import { useState, useCallback, type ReactNode } from "react";
import { ChevronRight, Folder } from "lucide-react";
import { cn } from "@/lib/utils";

interface TreeNode<T> {
  children: Map<string, TreeNode<T>>;
  items: T[];
}

function buildTrie<T>(items: T[], getPath: (item: T) => string): TreeNode<T> {
  const root: TreeNode<T> = { children: new Map(), items: [] };
  for (const item of items) {
    const path = getPath(item);
    if (!path) {
      root.items.push(item);
      continue;
    }
    const parts = path.split("/");
    let node = root;
    for (const part of parts) {
      if (!node.children.has(part)) {
        node.children.set(part, { children: new Map(), items: [] });
      }
      node = node.children.get(part)!;
    }
    node.items.push(item);
  }
  return root;
}

function countItems<T>(node: TreeNode<T>): number {
  let count = node.items.length;
  for (const child of node.children.values()) {
    count += countItems(child);
  }
  return count;
}

interface FolderNodeProps<T> {
  name: string;
  fullPath: string;
  node: TreeNode<T>;
  renderLeaf: (item: T) => ReactNode;
  expandState: Record<string, boolean>;
  onToggle: (path: string) => void;
}

function FolderNode<T>({ name, fullPath, node, renderLeaf, expandState, onToggle }: FolderNodeProps<T>) {
  const isOpen = expandState[fullPath] !== false; // default open
  const count = countItems(node);

  return (
    <div>
      <button
        onClick={() => onToggle(fullPath)}
        className="flex items-center gap-1 py-1 px-1 w-full text-left hover:bg-accent rounded text-sm"
      >
        <ChevronRight
          className={cn("h-3.5 w-3.5 shrink-0 transition-transform", isOpen && "rotate-90")}
        />
        <Folder className="h-3.5 w-3.5 shrink-0 text-muted-foreground" />
        <span className="font-medium">{name}</span>
        <span className="text-muted-foreground text-xs ml-1">({count})</span>
      </button>
      {isOpen && (
        <div className="ml-4 border-l pl-2">
          {renderChildren(node, renderLeaf, fullPath, expandState, onToggle)}
        </div>
      )}
    </div>
  );
}

function renderChildren<T>(
  node: TreeNode<T>,
  renderLeaf: (item: T) => ReactNode,
  parentPath: string,
  expandState: Record<string, boolean>,
  onToggle: (path: string) => void,
): ReactNode {
  const folders = Array.from(node.children.entries()).sort(([a], [b]) => a.localeCompare(b));
  return (
    <>
      {folders.map(([name, child]) => {
        const fp = parentPath ? `${parentPath}/${name}` : name;
        return (
          <FolderNode
            key={fp}
            name={name}
            fullPath={fp}
            node={child}
            renderLeaf={renderLeaf}
            expandState={expandState}
            onToggle={onToggle}
          />
        );
      })}
      {node.items.map((item, i) => (
        <div key={i}>{renderLeaf(item)}</div>
      ))}
    </>
  );
}

interface FolderTreeProps<T> {
  items: T[];
  getPath: (item: T) => string;
  renderLeaf: (item: T) => ReactNode;
  stateKey?: string;
}

// Persist expand state across re-renders using a module-level map
const expandStates: Record<string, Record<string, boolean>> = {};

export function FolderTree<T>({ items, getPath, renderLeaf, stateKey = "default" }: FolderTreeProps<T>) {
  if (!expandStates[stateKey]) {
    expandStates[stateKey] = {};
  }
  const [, setTick] = useState(0);

  const onToggle = useCallback(
    (path: string) => {
      const st = expandStates[stateKey];
      st[path] = st[path] === false ? true : false;
      setTick((t) => t + 1);
    },
    [stateKey],
  );

  const root = buildTrie(items, getPath);
  const expandState = expandStates[stateKey];

  // If no folders at all, just render items flat
  if (root.children.size === 0) {
    return <div className="space-y-0.5">{root.items.map((item, i) => <div key={i}>{renderLeaf(item)}</div>)}</div>;
  }

  return <div className="space-y-0.5">{renderChildren(root, renderLeaf, "", expandState, onToggle)}</div>;
}

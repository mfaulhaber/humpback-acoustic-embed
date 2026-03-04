import { useState, useCallback } from "react";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogFooter,
} from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Folder, ChevronRight, Loader2 } from "lucide-react";
import { useBrowseDirectories } from "@/hooks/queries/useClassifier";

interface FolderBrowserProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  onSelect: (path: string) => void;
  initialPath?: string;
}

export function FolderBrowser({
  open,
  onOpenChange,
  onSelect,
  initialPath = "/",
}: FolderBrowserProps) {
  const [currentPath, setCurrentPath] = useState(initialPath);
  const { data, isLoading, isError } = useBrowseDirectories(
    open ? currentPath : null,
  );

  const breadcrumbs = currentPath.split("/").filter(Boolean);

  const navigateTo = useCallback((path: string) => {
    setCurrentPath(path);
  }, []);

  const handleBreadcrumbClick = useCallback(
    (index: number) => {
      const path = "/" + breadcrumbs.slice(0, index + 1).join("/");
      setCurrentPath(path);
    },
    [breadcrumbs],
  );

  const handleSelect = () => {
    onSelect(currentPath);
    onOpenChange(false);
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-lg">
        <DialogHeader>
          <DialogTitle>Browse Folder</DialogTitle>
        </DialogHeader>

        {/* Breadcrumb navigation */}
        <div className="flex items-center gap-1 text-sm overflow-x-auto pb-1">
          <button
            className="text-blue-600 hover:underline shrink-0"
            onClick={() => setCurrentPath("/")}
          >
            /
          </button>
          {breadcrumbs.map((part, i) => (
            <span key={i} className="flex items-center gap-1 shrink-0">
              <ChevronRight className="h-3 w-3 text-muted-foreground" />
              <button
                className={
                  i === breadcrumbs.length - 1
                    ? "font-medium"
                    : "text-blue-600 hover:underline"
                }
                onClick={() => handleBreadcrumbClick(i)}
              >
                {part}
              </button>
            </span>
          ))}
        </div>

        {/* Directory list */}
        <ScrollArea className="h-64 border rounded-md">
          {isLoading && (
            <div className="flex items-center justify-center h-full">
              <Loader2 className="h-5 w-5 animate-spin text-muted-foreground" />
            </div>
          )}
          {isError && (
            <div className="p-3 text-sm text-red-600">
              Failed to load directory
            </div>
          )}
          {data && data.subdirectories.length === 0 && (
            <div className="p-3 text-sm text-muted-foreground">
              No subdirectories
            </div>
          )}
          {data &&
            data.subdirectories.map((dir) => (
              <button
                key={dir.path}
                className="flex items-center gap-2 w-full px-3 py-2 text-sm hover:bg-muted text-left"
                onClick={() => navigateTo(dir.path)}
              >
                <Folder className="h-4 w-4 text-muted-foreground shrink-0" />
                <span className="truncate">{dir.name}</span>
              </button>
            ))}
        </ScrollArea>

        <div className="text-xs text-muted-foreground truncate">
          Selected: {currentPath}
        </div>

        <DialogFooter>
          <Button variant="outline" onClick={() => onOpenChange(false)}>
            Cancel
          </Button>
          <Button onClick={handleSelect}>Select</Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}

import { useCallback, useEffect, useRef, useState, type ReactNode } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

export interface CollapsiblePanelCardProps {
  title: ReactNode;
  storageKey: string;
  defaultOpen?: boolean;
  headerExtra?: ReactNode;
  testId?: string;
  children: ReactNode;
}

const STORAGE_PREFIX = "seq-models:panel:";

function readPersisted(storageKey: string): boolean | null {
  try {
    const raw = window.localStorage.getItem(`${STORAGE_PREFIX}${storageKey}`);
    if (raw === "true") return true;
    if (raw === "false") return false;
  } catch {
    // ignore — SSR or denied storage
  }
  return null;
}

function writePersisted(storageKey: string, value: boolean): void {
  try {
    window.localStorage.setItem(`${STORAGE_PREFIX}${storageKey}`, String(value));
  } catch {
    // ignore — SSR or denied storage
  }
}

export function CollapsiblePanelCard({
  title,
  storageKey,
  defaultOpen = true,
  headerExtra,
  testId,
  children,
}: CollapsiblePanelCardProps) {
  const [open, setOpen] = useState<boolean>(() => {
    const persisted = readPersisted(storageKey);
    return persisted ?? defaultOpen;
  });
  const headerExtraRef = useRef<HTMLDivElement | null>(null);
  const resolvedTestId = testId ?? "collapsible-panel";
  const contentId = `${resolvedTestId}-content`;

  const toggle = useCallback(() => {
    setOpen((prev) => {
      const next = !prev;
      writePersisted(storageKey, next);
      return next;
    });
  }, [storageKey]);

  useEffect(() => {
    // Re-sync when the storageKey changes (e.g., a card is reused under a
    // different identity). New mount path normally handles this, but the
    // explicit re-read keeps the controlled state coherent if a parent
    // swaps keys without remounting.
    const persisted = readPersisted(storageKey);
    if (persisted !== null) setOpen(persisted);
  }, [storageKey]);

  const handleHeaderClick = (event: React.MouseEvent<HTMLDivElement>) => {
    if (
      headerExtraRef.current &&
      event.target instanceof Node &&
      headerExtraRef.current.contains(event.target)
    ) {
      return;
    }
    toggle();
  };

  return (
    <Card data-testid={resolvedTestId}>
      <CardHeader>
        <div
          className="flex cursor-pointer items-center justify-between"
          onClick={handleHeaderClick}
          role="presentation"
        >
          <div className="flex items-center gap-2">
            <button
              type="button"
              onClick={(event) => {
                event.stopPropagation();
                toggle();
              }}
              aria-expanded={open}
              aria-controls={contentId}
              data-testid={`${resolvedTestId}-toggle`}
              className="text-xs text-muted-foreground hover:text-foreground"
            >
              {open ? "▾" : "▸"}
            </button>
            <CardTitle className="text-base">{title}</CardTitle>
          </div>
          {headerExtra != null && (
            <div ref={headerExtraRef} data-testid={`${resolvedTestId}-header-extra`}>
              {headerExtra}
            </div>
          )}
        </div>
      </CardHeader>
      {open && <CardContent id={contentId}>{children}</CardContent>}
    </Card>
  );
}

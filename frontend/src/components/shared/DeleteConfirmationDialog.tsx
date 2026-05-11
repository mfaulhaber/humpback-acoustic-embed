import { useState, type ReactNode } from "react";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Button, type ButtonProps } from "@/components/ui/button";

type DeleteTarget =
  | {
      resourceName: ReactNode;
      count?: never;
    }
  | {
      resourceName?: never;
      count: number;
    };

interface DeleteConfirmationDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  resourceType: string;
  pluralResourceType?: string;
  confirmationText?: ReactNode;
  consequence: ReactNode;
  onConfirm: () => unknown | Promise<unknown>;
  isPending?: boolean;
}

export type DeleteConfirmationProps =
  DeleteConfirmationDialogProps & DeleteTarget;

function pluralize(resourceType: string): string {
  return resourceType.endsWith("s") ? resourceType : `${resourceType}s`;
}

function resourceLabel({
  count,
  resourceName,
  resourceType,
  pluralResourceType,
}: Pick<DeleteConfirmationProps, "count" | "resourceName" | "resourceType" | "pluralResourceType">) {
  if (typeof count === "number") {
    const label = count === 1 ? resourceType : (pluralResourceType ?? pluralize(resourceType));
    return (
      <>
        <strong className="font-semibold text-foreground">{count}</strong>{" "}
        {label}
      </>
    );
  }

  return (
    <>
      {resourceType}{" "}
      <strong className="font-semibold text-foreground">{resourceName}</strong>
    </>
  );
}

function titleFor({
  count,
  resourceType,
  pluralResourceType,
}: Pick<DeleteConfirmationProps, "count" | "resourceType" | "pluralResourceType">): string {
  if (typeof count === "number" && count !== 1) {
    return `Delete ${pluralResourceType ?? pluralize(resourceType)}`;
  }
  return `Delete ${resourceType}`;
}

export function DeleteActionButton({
  children = "Delete",
  variant = "delete",
  size = "sm",
  ...props
}: ButtonProps) {
  return (
    <Button variant={variant} size={size} {...props}>
      {children}
    </Button>
  );
}

export function DeleteConfirmationDialog({
  open,
  onOpenChange,
  resourceType,
  pluralResourceType,
  confirmationText,
  resourceName,
  count,
  consequence,
  onConfirm,
  isPending = false,
}: DeleteConfirmationProps) {
  const handleConfirm = async () => {
    await onConfirm();
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-md">
        <DialogHeader>
          <DialogTitle>
            {titleFor({ count, resourceType, pluralResourceType })}
          </DialogTitle>
          <DialogDescription asChild>
            <div className="space-y-3 text-sm text-muted-foreground">
              <p>
                {confirmationText ?? (
                  <>
                    Permanently delete{" "}
                    {resourceLabel({
                      count,
                      resourceName,
                      resourceType,
                      pluralResourceType,
                    })}
                    ? You can't undo this action.
                  </>
                )}
              </p>
              <div className="rounded-md border border-red-200 bg-red-50 px-3 py-2 text-red-900 dark:border-red-900/60 dark:bg-red-950/40 dark:text-red-200">
                {consequence}
              </div>
            </div>
          </DialogDescription>
        </DialogHeader>
        <DialogFooter>
          <Button
            variant="outline"
            onClick={() => onOpenChange(false)}
            disabled={isPending}
          >
            Cancel
          </Button>
          <DeleteActionButton onClick={handleConfirm} disabled={isPending}>
            {isPending ? "Deleting..." : "Delete"}
          </DeleteActionButton>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}

interface DeleteConfirmButtonProps extends Omit<ButtonProps, "onClick" | "variant"> {
  resourceType: string;
  pluralResourceType?: string;
  resourceName: ReactNode;
  confirmationText?: ReactNode;
  consequence: ReactNode;
  onConfirm: () => unknown | Promise<unknown>;
  isPending?: boolean;
}

export function DeleteConfirmButton({
  resourceType,
  pluralResourceType,
  resourceName,
  confirmationText,
  consequence,
  onConfirm,
  isPending = false,
  children = "Delete",
  disabled,
  ...buttonProps
}: DeleteConfirmButtonProps) {
  const [open, setOpen] = useState(false);

  const handleConfirm = async () => {
    await onConfirm();
    setOpen(false);
  };

  return (
    <>
      <DeleteActionButton
        {...buttonProps}
        disabled={disabled || isPending}
        onClick={() => setOpen(true)}
      >
        {children}
      </DeleteActionButton>
      <DeleteConfirmationDialog
        open={open}
        onOpenChange={setOpen}
        resourceType={resourceType}
        pluralResourceType={pluralResourceType}
        resourceName={resourceName}
        confirmationText={confirmationText}
        consequence={consequence}
        onConfirm={handleConfirm}
        isPending={isPending}
      />
    </>
  );
}

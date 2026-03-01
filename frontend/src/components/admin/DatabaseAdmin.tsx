import { useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import { useTables, useDeleteAllRecords } from "@/hooks/queries/useAdmin";
import { showMsg } from "@/components/shared/MessageToast";

export function DatabaseAdmin() {
  const { data: tables = [] } = useTables();
  const deleteAll = useDeleteAllRecords();
  const [confirmOpen, setConfirmOpen] = useState(false);

  return (
    <Card>
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <CardTitle className="text-base">Database Tables</CardTitle>
          <Dialog open={confirmOpen} onOpenChange={setConfirmOpen}>
            <DialogTrigger asChild>
              <Button variant="destructive" size="sm">
                Delete All Records
              </Button>
            </DialogTrigger>
            <DialogContent>
              <DialogHeader>
                <DialogTitle>Delete All Records</DialogTitle>
                <DialogDescription>
                  This will permanently delete all data from every table. This action cannot be undone.
                </DialogDescription>
              </DialogHeader>
              <DialogFooter>
                <Button variant="outline" onClick={() => setConfirmOpen(false)}>
                  Cancel
                </Button>
                <Button
                  variant="destructive"
                  onClick={() => {
                    deleteAll.mutate(undefined, {
                      onSuccess: () => {
                        showMsg("success", "All records deleted");
                        setConfirmOpen(false);
                      },
                      onError: (e) => showMsg("error", e.message),
                    });
                  }}
                  disabled={deleteAll.isPending}
                >
                  {deleteAll.isPending ? "Deleting..." : "Delete Everything"}
                </Button>
              </DialogFooter>
            </DialogContent>
          </Dialog>
        </div>
      </CardHeader>
      <CardContent>
        <div className="border rounded-md">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b bg-muted/50">
                <th className="text-left py-2 px-3 font-medium">Table</th>
                <th className="text-left py-2 px-3 font-medium">Row Count</th>
              </tr>
            </thead>
            <tbody>
              {tables.map((t) => (
                <tr key={t.table} className="border-b last:border-0">
                  <td className="py-1.5 px-3 font-mono text-xs">{t.table}</td>
                  <td className="py-1.5 px-3">{t.count}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </CardContent>
    </Card>
  );
}

import { useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Search } from "lucide-react";
import { useScanModels, useCreateModel } from "@/hooks/queries/useAdmin";
import { showMsg } from "@/components/shared/MessageToast";
import type { AvailableModelFile } from "@/api/types";

export function ModelScanner() {
  const scanMutation = useScanModels();
  const [scanned, setScanned] = useState<AvailableModelFile[]>([]);

  const handleScan = () => {
    scanMutation.mutate(undefined, {
      onSuccess: (data) => {
        const unregistered = data.filter((f) => !f.registered);
        setScanned(unregistered);
        if (unregistered.length === 0) {
          showMsg("success", "No unregistered model files found");
        }
      },
      onError: (e) => showMsg("error", `Scan failed: ${e.message}`),
    });
  };

  return (
    <Card>
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <CardTitle className="text-base">Model Scanner</CardTitle>
          <Button variant="outline" size="sm" onClick={handleScan} disabled={scanMutation.isPending}>
            <Search className="h-3.5 w-3.5 mr-1" />
            {scanMutation.isPending ? "Scanning..." : "Scan for Models"}
          </Button>
        </div>
      </CardHeader>
      {scanned.length > 0 && (
        <CardContent className="space-y-3">
          {scanned.map((file) => (
            <ModelFileCard key={file.path} file={file} onRegistered={() => setScanned((s) => s.filter((f) => f.path !== file.path))} />
          ))}
        </CardContent>
      )}
    </Card>
  );
}

function ModelFileCard({ file, onRegistered }: { file: AvailableModelFile; onRegistered: () => void }) {
  const createModel = useCreateModel();
  const [name, setName] = useState(file.filename.replace(/\.[^.]+$/, ""));
  const [displayName, setDisplayName] = useState(file.filename.replace(/\.[^.]+$/, ""));
  const [vectorDim, setVectorDim] = useState("1280");

  return (
    <div className="border rounded-md p-3 space-y-2">
      <p className="text-sm font-medium">{file.filename}</p>
      <p className="text-xs text-muted-foreground">{file.path} ({(file.size_bytes / 1024 / 1024).toFixed(1)} MB)</p>
      <div className="grid grid-cols-3 gap-2">
        <div>
          <label className="text-xs text-muted-foreground">Name</label>
          <Input value={name} onChange={(e) => setName(e.target.value)} className="h-8" />
        </div>
        <div>
          <label className="text-xs text-muted-foreground">Display Name</label>
          <Input value={displayName} onChange={(e) => setDisplayName(e.target.value)} className="h-8" />
        </div>
        <div>
          <label className="text-xs text-muted-foreground">Vector Dim</label>
          <Input type="number" value={vectorDim} onChange={(e) => setVectorDim(e.target.value)} className="h-8" />
        </div>
      </div>
      <Button
        size="sm"
        onClick={() =>
          createModel.mutate(
            {
              name,
              display_name: displayName,
              path: file.path,
              vector_dim: parseInt(vectorDim) || 1280,
              model_type: file.model_type,
              input_format: file.input_format,
            },
            {
              onSuccess: () => {
                showMsg("success", `Model "${name}" registered`);
                onRegistered();
              },
              onError: (e) => showMsg("error", e.message),
            },
          )
        }
        disabled={createModel.isPending}
      >
        Register
      </Button>
    </div>
  );
}

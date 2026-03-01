import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { useModels, useSetDefaultModel, useDeleteModel } from "@/hooks/queries/useAdmin";
import { showMsg } from "@/components/shared/MessageToast";

export function ModelRegistry() {
  const { data: models = [] } = useModels();
  const setDefault = useSetDefaultModel();
  const deleteModel = useDeleteModel();

  return (
    <Card>
      <CardHeader className="pb-3">
        <CardTitle className="text-base">Model Registry</CardTitle>
      </CardHeader>
      <CardContent>
        {models.length === 0 ? (
          <p className="text-sm text-muted-foreground">No models registered.</p>
        ) : (
          <div className="border rounded-md">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b bg-muted/50">
                  <th className="text-left py-2 px-3 font-medium">Name</th>
                  <th className="text-left py-2 px-3 font-medium">Display Name</th>
                  <th className="text-left py-2 px-3 font-medium">Path</th>
                  <th className="text-left py-2 px-3 font-medium">Type</th>
                  <th className="text-left py-2 px-3 font-medium">Format</th>
                  <th className="text-left py-2 px-3 font-medium">Dims</th>
                  <th className="text-left py-2 px-3 font-medium">Default</th>
                  <th className="text-left py-2 px-3 font-medium">Actions</th>
                </tr>
              </thead>
              <tbody>
                {models.map((model) => (
                  <tr key={model.id} className="border-b last:border-0">
                    <td className="py-2 px-3 font-mono text-xs">{model.name}</td>
                    <td className="py-2 px-3">{model.display_name}</td>
                    <td className="py-2 px-3 text-xs text-muted-foreground truncate max-w-[200px]">
                      {model.path}
                    </td>
                    <td className="py-2 px-3 text-xs">{model.model_type}</td>
                    <td className="py-2 px-3 text-xs">{model.input_format}</td>
                    <td className="py-2 px-3">{model.vector_dim}</td>
                    <td className="py-2 px-3">
                      {model.is_default && <Badge variant="secondary">Default</Badge>}
                    </td>
                    <td className="py-2 px-3">
                      <div className="flex gap-1">
                        {!model.is_default && (
                          <Button
                            variant="outline"
                            size="sm"
                            className="h-7 text-xs"
                            onClick={() =>
                              setDefault.mutate(model.id, {
                                onSuccess: () => showMsg("success", "Default model updated"),
                                onError: (e) => showMsg("error", e.message),
                              })
                            }
                          >
                            Set Default
                          </Button>
                        )}
                        <Button
                          variant="destructive"
                          size="sm"
                          className="h-7 text-xs"
                          onClick={() =>
                            deleteModel.mutate(model.id, {
                              onSuccess: () => showMsg("success", "Model deleted"),
                              onError: (e) => showMsg("error", e.message),
                            })
                          }
                        >
                          Delete
                        </Button>
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </CardContent>
    </Card>
  );
}

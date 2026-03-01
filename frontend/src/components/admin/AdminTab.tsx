import { ModelRegistry } from "./ModelRegistry";
import { ModelScanner } from "./ModelScanner";
import { DatabaseAdmin } from "./DatabaseAdmin";

export function AdminTab() {
  return (
    <div className="space-y-4">
      <ModelRegistry />
      <ModelScanner />
      <DatabaseAdmin />
    </div>
  );
}

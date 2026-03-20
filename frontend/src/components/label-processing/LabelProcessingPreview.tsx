import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import type { LabelProcessingPreview as PreviewData } from "@/api/types";

interface Props {
  data: PreviewData;
}

export function LabelProcessingPreview({ data }: Props) {
  const sortedCallTypes = Object.entries(data.call_type_distribution).sort(
    ([, a], [, b]) => b - a,
  );
  const maxCount = sortedCallTypes.length > 0 ? sortedCallTypes[0][1] : 1;

  return (
    <div className="space-y-4">
      {/* Summary stats */}
      <div className="flex gap-6 text-sm">
        <div>
          <span className="text-muted-foreground">Paired files:</span>{" "}
          <span className="font-medium">{data.paired_files.length}</span>
        </div>
        <div>
          <span className="text-muted-foreground">Total annotations:</span>{" "}
          <span className="font-medium">{data.total_annotations}</span>
        </div>
        {data.unpaired_annotations.length > 0 && (
          <div className="text-amber-600">
            {data.unpaired_annotations.length} unpaired annotation file(s)
          </div>
        )}
        {data.unpaired_audio.length > 0 && (
          <div className="text-amber-600">
            {data.unpaired_audio.length} unpaired audio file(s)
          </div>
        )}
      </div>

      {/* Call type distribution */}
      {sortedCallTypes.length > 0 && (
        <Card>
          <CardHeader className="py-3 px-4">
            <CardTitle className="text-sm">Call Type Distribution</CardTitle>
          </CardHeader>
          <CardContent className="px-4 pb-3 pt-0">
            <div className="space-y-1.5">
              {sortedCallTypes.map(([callType, count]) => (
                <div key={callType} className="flex items-center gap-3 text-sm">
                  <span className="w-36 truncate text-muted-foreground" title={callType}>
                    {callType}
                  </span>
                  <div className="flex-1 h-4 bg-slate-100 rounded overflow-hidden">
                    <div
                      className="h-full bg-blue-400 rounded"
                      style={{ width: `${(count / maxCount) * 100}%` }}
                    />
                  </div>
                  <span className="w-10 text-right font-mono text-xs">{count}</span>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}

      {/* Paired file table */}
      <Card>
        <CardHeader className="py-3 px-4">
          <CardTitle className="text-sm">File Pairing</CardTitle>
        </CardHeader>
        <CardContent className="px-4 pb-3 pt-0">
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b text-left text-muted-foreground">
                  <th className="py-1.5 pr-4 font-medium">Annotation File</th>
                  <th className="py-1.5 pr-4 font-medium">Audio File</th>
                  <th className="py-1.5 font-medium text-right">Annotations</th>
                </tr>
              </thead>
              <tbody>
                {data.paired_files.map((pf) => (
                  <tr key={pf.annotation_file} className="border-b last:border-0">
                    <td className="py-1.5 pr-4 font-mono text-xs truncate max-w-[300px]">
                      {pf.annotation_file}
                    </td>
                    <td className="py-1.5 pr-4 font-mono text-xs truncate max-w-[300px]">
                      {pf.audio_file}
                    </td>
                    <td className="py-1.5 text-right">{pf.annotation_count}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </CardContent>
      </Card>

      {/* Unpaired files if any */}
      {(data.unpaired_annotations.length > 0 || data.unpaired_audio.length > 0) && (
        <Card>
          <CardHeader className="py-3 px-4">
            <CardTitle className="text-sm text-amber-600">Unpaired Files</CardTitle>
          </CardHeader>
          <CardContent className="px-4 pb-3 pt-0 text-sm space-y-2">
            {data.unpaired_annotations.length > 0 && (
              <div>
                <div className="text-muted-foreground mb-1">Annotations without audio:</div>
                <ul className="list-disc list-inside space-y-0.5">
                  {data.unpaired_annotations.map((f) => (
                    <li key={f} className="font-mono text-xs truncate">{f}</li>
                  ))}
                </ul>
              </div>
            )}
            {data.unpaired_audio.length > 0 && (
              <div>
                <div className="text-muted-foreground mb-1">Audio without annotations:</div>
                <ul className="list-disc list-inside space-y-0.5">
                  {data.unpaired_audio.map((f) => (
                    <li key={f} className="font-mono text-xs truncate">{f}</li>
                  ))}
                </ul>
              </div>
            )}
          </CardContent>
        </Card>
      )}
    </div>
  );
}

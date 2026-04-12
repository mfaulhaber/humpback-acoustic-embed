interface RegionJobSummaryProps {
  regionCount: number | null;
  highThreshold: number | null;
  lowThreshold: number | null;
}

export function RegionJobSummary({
  regionCount,
  highThreshold,
  lowThreshold,
}: RegionJobSummaryProps) {
  return (
    <span className="text-xs text-muted-foreground">
      {regionCount != null ? `${regionCount} region${regionCount !== 1 ? "s" : ""}` : "—"}
      {highThreshold != null && lowThreshold != null && (
        <span className="ml-2">
          ({highThreshold.toFixed(2)}/{lowThreshold.toFixed(2)})
        </span>
      )}
    </span>
  );
}

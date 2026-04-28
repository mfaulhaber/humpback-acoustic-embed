export function regionTileIndexForSpanTile(
  activeSpanStartTimestamp: number,
  regionStartTimestamp: number,
  spanTileIndex: number,
  tileDurationSec: number,
): number {
  const tileStartTimestamp =
    activeSpanStartTimestamp + spanTileIndex * tileDurationSec;
  return Math.max(
    0,
    Math.floor(
      (tileStartTimestamp - regionStartTimestamp) / tileDurationSec,
    ),
  );
}

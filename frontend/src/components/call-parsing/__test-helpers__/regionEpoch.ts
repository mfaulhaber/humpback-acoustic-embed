// 2021-10-31T00:00:00Z — non-zero epoch base used by call-parsing
// review workspace tests so a relative→epoch conversion bug cannot
// silently pass against a zero-anchored coordinate space.
export const REGION_EPOCH_BASE = 1635638400;
export const REGION_EPOCH_END = REGION_EPOCH_BASE + 86400;

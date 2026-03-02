import { useCallback } from "react";
import { useSearchParams } from "react-router-dom";

/**
 * URL-persisted collapse state. Stores expanded keys as a comma-separated list
 * in a search param. Everything is collapsed by default.
 *
 * @param stateKey  Prefix to namespace keys within the same param (e.g. "audioTree")
 * @param paramName The URL search-param name (e.g. "ft" for folder-tree, "cj" for clustering-jobs)
 */
export function useCollapseState(stateKey: string, paramName: string) {
  const [searchParams, setSearchParams] = useSearchParams();

  const expandedKeys = parseParam(searchParams.get(paramName), stateKey);

  const isExpanded = useCallback(
    (key: string): boolean => expandedKeys.has(key),
    [expandedKeys],
  );

  const toggle = useCallback(
    (key: string) => {
      setSearchParams(
        (prev) => {
          const current = parseParam(prev.get(paramName), stateKey);
          if (current.has(key)) {
            current.delete(key);
          } else {
            current.add(key);
          }
          const next = new URLSearchParams(prev);
          const serialized = serializeParam(next.get(paramName), stateKey, current);
          if (serialized) {
            next.set(paramName, serialized);
          } else {
            next.delete(paramName);
          }
          return next;
        },
        { replace: true },
      );
    },
    [paramName, stateKey, setSearchParams],
  );

  return { isExpanded, toggle };
}

/** Parse "stateKey:path1,stateKey:path2,otherKey:x" → Set of paths for stateKey */
function parseParam(raw: string | null, stateKey: string): Set<string> {
  const result = new Set<string>();
  if (!raw) return result;
  const prefix = stateKey + ":";
  for (const entry of raw.split(",")) {
    if (entry.startsWith(prefix)) {
      result.add(entry.slice(prefix.length));
    }
  }
  return result;
}

/** Serialize: merge updated keys for stateKey back into the full param value */
function serializeParam(raw: string | null, stateKey: string, keys: Set<string>): string {
  const prefix = stateKey + ":";
  // Keep entries from other stateKeys
  const others: string[] = [];
  if (raw) {
    for (const entry of raw.split(",")) {
      if (!entry.startsWith(prefix) && entry.length > 0) {
        others.push(entry);
      }
    }
  }
  // Add entries for this stateKey
  for (const key of keys) {
    others.push(prefix + key);
  }
  return others.join(",");
}

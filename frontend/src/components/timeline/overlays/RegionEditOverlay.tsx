import React, { useState, useCallback, useRef } from "react";
import type { Region, RegionCorrection } from "@/api/types";
import { useOverlayContext } from "./OverlayContext";

interface EditableRegion {
  region_id: string;
  start_sec: number;
  end_sec: number;
  max_score: number;
  correctionType: "adjust" | "add" | "delete" | null;
}

interface RegionEditOverlayProps {
  regions: Region[];
  corrections: Map<string, RegionCorrection>;
  jobStart: number;
  addMode?: boolean;
  selectedRegionId: string | null;
  onSelectRegion: (regionId: string | null) => void;
  onCorrection: (correction: RegionCorrection) => void;
}

export function RegionEditOverlay({
  regions,
  corrections,
  jobStart,
  addMode = false,
  selectedRegionId,
  onSelectRegion,
  onCorrection,
}: RegionEditOverlayProps) {
  const { epochToX, xToEpoch, canvasWidth, canvasHeight, pxPerSec } = useOverlayContext();

  const [dragState, setDragState] = useState<{
    regionId: string;
    edge: "start" | "end";
    initialSec: number;
    initialX: number;
  } | null>(null);

  const [newRegionDrag, setNewRegionDrag] = useState<{
    startX: number;
    currentX: number;
  } | null>(null);

  const containerRef = useRef<HTMLDivElement>(null);

  const xToJobRelativeSec = useCallback(
    (x: number) => xToEpoch(x) - jobStart,
    [xToEpoch, jobStart],
  );

  const editableRegions: EditableRegion[] = regions
    .filter((r) => {
      const c = corrections.get(r.region_id);
      return !c || c.correction_type !== "delete";
    })
    .map((r) => {
      const c = corrections.get(r.region_id);
      if (c && c.correction_type === "adjust") {
        return {
          region_id: r.region_id,
          start_sec: c.start_sec!,
          end_sec: c.end_sec!,
          max_score: r.max_score,
          correctionType: "adjust" as const,
        };
      }
      return {
        region_id: r.region_id,
        start_sec: r.padded_start_sec,
        end_sec: r.padded_end_sec,
        max_score: r.max_score,
        correctionType: null,
      };
    });

  corrections.forEach((c, regionId) => {
    if (c.correction_type === "add" && !regions.some((r) => r.region_id === regionId)) {
      editableRegions.push({
        region_id: regionId,
        start_sec: c.start_sec!,
        end_sec: c.end_sec!,
        max_score: 0,
        correctionType: "add",
      });
    }
  });

  const handleEdgeDragStart = useCallback(
    (e: React.MouseEvent, regionId: string, edge: "start" | "end", sec: number) => {
      e.stopPropagation();
      onSelectRegion(regionId);
      setDragState({ regionId, edge, initialSec: sec, initialX: e.clientX });
    },
    [onSelectRegion],
  );

  const handleMouseMove = useCallback(
    (e: React.MouseEvent) => {
      e.stopPropagation();
      if (dragState) {
        const dx = e.clientX - dragState.initialX;
        const deltaSec = dx / pxPerSec;
        const newSec = Math.max(0, dragState.initialSec + deltaSec);

        const region = editableRegions.find((r) => r.region_id === dragState.regionId);
        if (!region) return;

        const newStart = dragState.edge === "start" ? newSec : region.start_sec;
        const newEnd = dragState.edge === "end" ? newSec : region.end_sec;

        if (newEnd > newStart) {
          onCorrection({
            region_id: dragState.regionId,
            correction_type: "adjust",
            start_sec: newStart,
            end_sec: newEnd,
          });
        }
      } else if (newRegionDrag) {
        setNewRegionDrag({ ...newRegionDrag, currentX: e.clientX });
      }
    },
    [dragState, newRegionDrag, pxPerSec, editableRegions, onCorrection],
  );

  const handleMouseUp = useCallback(
    (e: React.MouseEvent) => {
      e.stopPropagation();
      if (newRegionDrag && containerRef.current) {
        const rect = containerRef.current.getBoundingClientRect();
        const startSec = xToJobRelativeSec(newRegionDrag.startX - rect.left);
        const endSec = xToJobRelativeSec(newRegionDrag.currentX - rect.left);
        const lo = Math.max(0, Math.min(startSec, endSec));
        const hi = Math.max(startSec, endSec);
        if (hi - lo > 0.5) {
          const id = `add-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
          onCorrection({ region_id: id, correction_type: "add", start_sec: lo, end_sec: hi });
          onSelectRegion(id);
        }
      }
      setDragState(null);
      setNewRegionDrag(null);
    },
    [newRegionDrag, xToJobRelativeSec, onCorrection, onSelectRegion],
  );

  const handleBackgroundMouseDown = useCallback(
    (e: React.MouseEvent) => {
      if (e.button !== 0) return;
      e.stopPropagation();
      const target = e.target as HTMLElement;
      if (target.dataset.handle || target.dataset.region || target.dataset.delete) return;
      onSelectRegion(null);
      if (addMode) {
        setNewRegionDrag({ startX: e.clientX, currentX: e.clientX });
      }
    },
    [onSelectRegion, addMode],
  );

  const handleRegionClick = useCallback(
    (e: React.MouseEvent, regionId: string) => {
      e.stopPropagation();
      onSelectRegion(regionId);
    },
    [onSelectRegion],
  );

  const handleDelete = useCallback(
    (regionId: string) => {
      onCorrection({ region_id: regionId, correction_type: "delete", start_sec: null, end_sec: null });
      onSelectRegion(null);
    },
    [onCorrection, onSelectRegion],
  );

  const newRegionPreview = (() => {
    if (!newRegionDrag || !containerRef.current) return null;
    const rect = containerRef.current.getBoundingClientRect();
    const x1 = newRegionDrag.startX - rect.left;
    const x2 = newRegionDrag.currentX - rect.left;
    const left = Math.min(x1, x2);
    const w = Math.abs(x2 - x1);
    return { left, w };
  })();

  return (
    <div
      ref={containerRef}
      data-testid="region-edit-overlay"
      style={{
        position: "absolute",
        top: 0,
        left: 0,
        width: canvasWidth,
        height: canvasHeight,
        zIndex: 6,
        overflow: "hidden",
        cursor: newRegionDrag ? "crosshair" : dragState ? "col-resize" : addMode ? "crosshair" : "default",
      }}
      onMouseMove={handleMouseMove}
      onMouseUp={handleMouseUp}
      onMouseLeave={handleMouseUp}
      onMouseDown={handleBackgroundMouseDown}
    >
      {editableRegions.map((r) => {
        const startEpoch = jobStart + r.start_sec;
        const endEpoch = jobStart + r.end_sec;
        const x = epochToX(startEpoch);
        const w = Math.max(4, epochToX(endEpoch) - x);

        if (x + w < 0 || x > canvasWidth) return null;

        const isSelected = r.region_id === selectedRegionId;
        const isModified = r.correctionType !== null;
        const bgColor = isSelected
          ? "rgba(100, 180, 255, 0.30)"
          : isModified
            ? "rgba(255, 200, 64, 0.25)"
            : "rgba(64, 224, 192, 0.20)";
        const borderColor = isSelected
          ? "rgba(100, 180, 255, 1.0)"
          : isModified
            ? "rgba(255, 200, 64, 0.8)"
            : "rgba(64, 224, 192, 0.6)";

        return (
          <div
            key={r.region_id}
            data-region="true"
            style={{
              position: "absolute",
              top: 0,
              left: x,
              width: w,
              height: canvasHeight,
              background: bgColor,
              borderLeft: `2px solid ${borderColor}`,
              borderRight: `2px solid ${borderColor}`,
              boxShadow: isSelected ? `inset 0 0 0 1px ${borderColor}` : undefined,
              cursor: "pointer",
            }}
            onMouseDown={(e) => {
              e.stopPropagation();
              handleRegionClick(e, r.region_id);
            }}
          >
            <div
              data-handle="true"
              style={{ position: "absolute", left: -4, top: 0, width: 8, height: canvasHeight, cursor: "col-resize" }}
              onMouseDown={(e) => handleEdgeDragStart(e, r.region_id, "start", r.start_sec)}
            />
            <div
              data-handle="true"
              style={{ position: "absolute", right: -4, top: 0, width: 8, height: canvasHeight, cursor: "col-resize" }}
              onMouseDown={(e) => handleEdgeDragStart(e, r.region_id, "end", r.end_sec)}
            />
            {isSelected && (
              <button
                data-delete="true"
                style={{
                  position: "absolute",
                  top: 4,
                  right: 4,
                  padding: "2px 6px",
                  borderRadius: 3,
                  background: "rgba(239, 68, 68, 0.9)",
                  color: "#fff",
                  border: "none",
                  fontSize: 10,
                  fontWeight: 500,
                  cursor: "pointer",
                  zIndex: 2,
                }}
                onMouseDown={(e) => e.stopPropagation()}
                onClick={(e) => {
                  e.stopPropagation();
                  handleDelete(r.region_id);
                }}
                title="Delete region"
              >
                Delete
              </button>
            )}
          </div>
        );
      })}

      {newRegionPreview && (
        <div
          style={{
            position: "absolute",
            top: 0,
            left: newRegionPreview.left,
            width: newRegionPreview.w,
            height: canvasHeight,
            background: "rgba(120, 180, 255, 0.3)",
            border: "2px dashed rgba(120, 180, 255, 0.8)",
            pointerEvents: "none",
          }}
        />
      )}
    </div>
  );
}

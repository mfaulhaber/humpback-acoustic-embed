import { useState, useCallback, useEffect } from "react";
import { useParams, useNavigate } from "react-router-dom";
import { ArrowLeft, Plus } from "lucide-react";
import { Button } from "@/components/ui/button";
import type { RegionCorrection } from "@/api/types";
import { regionTileUrl, regionAudioSliceUrl } from "@/api/client";
import {
  useRegionDetectionJobs,
  useRegionJobRegions,
  useRegionJobConfidence,
  useRegionCorrections,
  useSaveRegionCorrections,
} from "@/hooks/queries/useCallParsing";
import { TimelineProvider } from "@/components/timeline/provider/TimelineProvider";
import { FULL_ZOOM } from "@/components/timeline/provider/types";
import { Spectrogram } from "@/components/timeline/spectrogram/Spectrogram";
import { RegionOverlay } from "@/components/timeline/overlays/RegionOverlay";
import { RegionEditOverlay } from "@/components/timeline/overlays/RegionEditOverlay";
import { ZoomSelector } from "@/components/timeline/controls/ZoomSelector";
import { PlaybackControls } from "@/components/timeline/controls/PlaybackControls";
import { EditToggle } from "@/components/timeline/controls/EditToggle";
import { EditToolbar } from "@/components/timeline/controls/EditToolbar";
import { OverlayToggles } from "@/components/timeline/controls/OverlayToggles";
import { TimelineFooter } from "@/components/timeline/controls/TimelineFooter";
import { COLORS } from "@/components/timeline/constants";

function tileUrlBuilder(
  jobId: string,
  zoomLevel: string,
  tileIndex: number,
  _freqMin: number,
  _freqMax: number,
): string {
  return regionTileUrl(jobId, zoomLevel, tileIndex);
}

export function RegionDetectionTimeline() {
  const { jobId } = useParams<{ jobId: string }>();
  const navigate = useNavigate();
  const { data: jobs } = useRegionDetectionJobs(0);
  const job = jobs?.find((j) => j.id === jobId);
  const { data: regions } = useRegionJobRegions(jobId ?? null);
  const { data: confidence } = useRegionJobConfidence(jobId ?? null);
  const { data: savedCorrections } = useRegionCorrections(jobId ?? null);
  const saveCorrections = useSaveRegionCorrections();

  const [editMode, setEditMode] = useState(false);
  const [addMode, setAddMode] = useState(false);
  const [showRegionOverlay, setShowRegionOverlay] = useState(true);
  const [selectedRegionId, setSelectedRegionId] = useState<string | null>(null);
  const [pendingCorrections, setPendingCorrections] = useState<Map<string, RegionCorrection>>(new Map());

  useEffect(() => {
    if (editMode && savedCorrections) {
      const map = new Map<string, RegionCorrection>();
      for (const c of savedCorrections) {
        map.set(c.region_id, {
          region_id: c.region_id,
          correction_type: c.correction_type,
          start_sec: c.start_sec,
          end_sec: c.end_sec,
        });
      }
      setPendingCorrections(map);
    }
  }, [editMode, savedCorrections]);

  const handleCorrection = useCallback((correction: RegionCorrection) => {
    setPendingCorrections((prev) => {
      const next = new Map(prev);
      next.set(correction.region_id, correction);
      return next;
    });
    if (correction.correction_type === "add") setAddMode(false);
  }, []);

  const handleSaveCorrections = useCallback(() => {
    if (!jobId) return;
    const corrections = Array.from(pendingCorrections.values());
    saveCorrections.mutate(
      { jobId, corrections },
      {
        onSuccess: () => {
          setEditMode(false);
          setPendingCorrections(new Map());
          setSelectedRegionId(null);
        },
      },
    );
  }, [jobId, pendingCorrections, saveCorrections]);

  const handleCancelEdit = useCallback(() => {
    setEditMode(false);
    setAddMode(false);
    setPendingCorrections(new Map());
    setSelectedRegionId(null);
  }, []);

  const handleRegionClick = useCallback(
    (regionId: string) => {
      if (job?.status === "complete") {
        setEditMode(true);
        setSelectedRegionId(regionId);
      }
    },
    [job?.status],
  );

  const toggleEditMode = useCallback(() => {
    if (editMode) handleCancelEdit();
    else setEditMode(true);
  }, [editMode, handleCancelEdit]);

  if (!jobId || !job) {
    return (
      <div
        className="fixed left-60 flex items-center justify-center"
        style={{ top: "3rem", right: 0, bottom: 0, background: COLORS.bg, color: COLORS.text, zIndex: 40 }}
      >
        Loading...
      </div>
    );
  }

  const jobStart = job.start_timestamp ?? 0;
  const jobEnd = job.end_timestamp ?? 0;
  const regionCount = regions?.length ?? job.region_count ?? 0;
  const startStr = jobStart
    ? new Date(jobStart * 1000).toISOString().slice(0, 16).replace("T", " ") + " UTC"
    : "";
  const endStr = jobEnd
    ? new Date(jobEnd * 1000).toISOString().slice(0, 16).replace("T", " ") + " UTC"
    : "";

  const audioUrlBuilder = (startEpoch: number, durationSec: number) => {
    const jobRelative = startEpoch - jobStart;
    return regionAudioSliceUrl(jobId, Math.max(0, jobRelative), durationSec);
  };

  return (
    <div
      className="fixed left-60 flex flex-col font-mono text-xs overflow-hidden"
      style={{ top: "3rem", right: 0, bottom: 0, background: COLORS.bg, color: COLORS.text, zIndex: 40 }}
    >
      {/* Header */}
      <div
        className="flex items-center gap-3 px-4 py-2 shrink-0"
        style={{ background: COLORS.headerBg, borderBottom: `1px solid ${COLORS.border}` }}
      >
        <Button variant="ghost" size="sm" onClick={() => navigate("/app/call-parsing/detection")}>
          <ArrowLeft className="h-3.5 w-3.5 mr-1" />
          Back
        </Button>
        <span style={{ color: COLORS.text, fontWeight: 600 }}>Region Detection Timeline</span>
        <span className="truncate min-w-0" style={{ color: COLORS.textMuted }}>
          {startStr} — {endStr}
        </span>
        <span className="shrink-0" style={{ color: COLORS.accent }}>
          {regionCount} region{regionCount !== 1 ? "s" : ""}
        </span>
      </div>

      {/* Timeline */}
      <TimelineProvider
        jobStart={jobStart}
        jobEnd={jobEnd}
        zoomLevels={FULL_ZOOM}
        defaultZoom="1h"
        playback="gapless"
        audioUrlBuilder={audioUrlBuilder}
      >
        <div className="flex-1 flex flex-col mx-4 my-2 rounded overflow-hidden min-h-0" style={{ background: COLORS.bgDark }}>
          <Spectrogram
            jobId={jobId}
            tileUrlBuilder={tileUrlBuilder}
            freqRange={[0, 3000]}
            scores={confidence?.scores}
            windowSec={confidence?.window_sec}
          >
            {showRegionOverlay && !editMode && regions && (
              <RegionOverlay
                regions={regions}
                jobStart={jobStart}
                visible={true}
                corrections={savedCorrections}
                onRegionClick={handleRegionClick}
              />
            )}
            {editMode && regions && (
              <RegionEditOverlay
                regions={regions}
                corrections={pendingCorrections}
                jobStart={jobStart}
                addMode={addMode}
                selectedRegionId={selectedRegionId}
                onSelectRegion={setSelectedRegionId}
                onCorrection={handleCorrection}
              />
            )}
          </Spectrogram>
        </div>

        {/* Footer */}
        <TimelineFooter>
          <div className="flex items-center">
            <div className="flex-1" />
            <ZoomSelector />
            <div className="flex-1 flex justify-end gap-2">
              {editMode && (
                <EditToolbar
                  pendingCount={pendingCorrections.size}
                  onSave={handleSaveCorrections}
                  onCancel={handleCancelEdit}
                  isSaving={saveCorrections.isPending}
                >
                  <button
                    className="flex items-center gap-1 px-2 py-0.5 rounded text-[10px] font-medium"
                    style={{
                      background: addMode ? "rgba(100, 180, 255, 0.3)" : "transparent",
                      color: addMode ? "rgba(100, 180, 255, 1)" : COLORS.accent,
                      border: `1px solid ${addMode ? "rgba(100, 180, 255, 0.8)" : COLORS.accent}`,
                    }}
                    onClick={() => setAddMode((v) => !v)}
                  >
                    <Plus size={10} /> Add
                  </button>
                </EditToolbar>
              )}
              <OverlayToggles
                options={[{ key: "regions", label: "Regions", active: showRegionOverlay }]}
                onToggle={() => setShowRegionOverlay((v) => !v)}
              />
            </div>
          </div>
          <PlaybackControls>
            <EditToggle
              active={editMode}
              enabled={job.status === "complete"}
              label="Edit Regions"
              onToggle={toggleEditMode}
            />
          </PlaybackControls>
        </TimelineFooter>
      </TimelineProvider>
    </div>
  );
}

import { Link, useLocation, useParams } from "react-router-dom";
import { useQueryClient } from "@tanstack/react-query";
import {
  Breadcrumb,
  BreadcrumbItem,
  BreadcrumbLink,
  BreadcrumbList,
  BreadcrumbPage,
  BreadcrumbSeparator,
} from "@/components/ui/breadcrumb";
import type { AudioFile } from "@/api/types";

interface Crumb {
  label: string;
  to?: string;
}

const staticRoutes: Record<string, Crumb[]> = {
  "/app/audio": [{ label: "Audio" }],
  "/app/processing": [{ label: "Processing" }],
  "/app/clustering": [{ label: "Clustering" }],
  "/app/classifier/training": [
    { label: "Classifier", to: "/app/classifier/training" },
    { label: "Training" },
  ],
  "/app/classifier/hydrophone": [
    { label: "Classifier", to: "/app/classifier/training" },
    { label: "Hydrophone Detection" },
  ],
  "/app/search": [{ label: "Search" }],
  "/app/admin": [{ label: "Admin" }],
};

function useAudioFilename(audioId: string | undefined): string | null {
  const queryClient = useQueryClient();
  if (!audioId) return null;

  // Try the list cache first
  const files = queryClient.getQueryData<AudioFile[]>(["audioFiles"]);
  const match = files?.find((f) => f.id === audioId);
  if (match) return match.filename;

  // Try individual file cache
  const single = queryClient.getQueryData<AudioFile>(["audioFile", audioId]);
  if (single) return single.filename;

  // Short ID fallback
  return audioId.slice(0, 8);
}

export function Breadcrumbs() {
  const { pathname } = useLocation();
  const { audioId, jobId } = useParams<{ audioId?: string; jobId?: string }>();
  const audioFilename = useAudioFilename(audioId);

  let crumbs: Crumb[];

  if (audioId && pathname.startsWith("/app/audio/")) {
    crumbs = [
      { label: "Audio", to: "/app/audio" },
      { label: audioFilename ?? audioId.slice(0, 8) },
    ];
  } else if (jobId && pathname.startsWith("/app/clustering/")) {
    crumbs = [
      { label: "Clustering", to: "/app/clustering" },
      { label: `Job ${jobId.slice(0, 8)}` },
    ];
  } else {
    crumbs = staticRoutes[pathname] ?? [{ label: "Audio" }];
  }

  if (crumbs.length === 0) return null;

  return (
    <Breadcrumb className="mb-4">
      <BreadcrumbList>
        {crumbs.map((crumb, i) => {
          const isLast = i === crumbs.length - 1;
          return (
            <BreadcrumbItem key={crumb.label}>
              {i > 0 && <BreadcrumbSeparator />}
              {isLast ? (
                <BreadcrumbPage>{crumb.label}</BreadcrumbPage>
              ) : (
                <BreadcrumbLink asChild>
                  <Link to={crumb.to!}>{crumb.label}</Link>
                </BreadcrumbLink>
              )}
            </BreadcrumbItem>
          );
        })}
      </BreadcrumbList>
    </Breadcrumb>
  );
}

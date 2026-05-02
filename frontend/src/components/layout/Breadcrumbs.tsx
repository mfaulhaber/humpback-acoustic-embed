import { Link, useLocation, useParams } from "react-router-dom";
import {
  Breadcrumb,
  BreadcrumbItem,
  BreadcrumbLink,
  BreadcrumbList,
  BreadcrumbPage,
  BreadcrumbSeparator,
} from "@/components/ui/breadcrumb";

interface Crumb {
  label: string;
  to?: string;
}

const staticRoutes: Record<string, Crumb[]> = {
  "/app/classifier/training": [
    { label: "Classifier", to: "/app/classifier/training" },
    { label: "Training" },
  ],
  "/app/classifier/hydrophone": [
    { label: "Classifier", to: "/app/classifier/training" },
    { label: "Hydrophone Detection" },
  ],
  "/app/call-parsing/detection": [
    { label: "Call Parsing", to: "/app/call-parsing" },
    { label: "Detection" },
  ],
  "/app/call-parsing/segment": [
    { label: "Call Parsing", to: "/app/call-parsing" },
    { label: "Segment" },
  ],
  "/app/call-parsing/segment-training": [
    { label: "Call Parsing", to: "/app/call-parsing" },
    { label: "Segment Training" },
  ],
  "/app/call-parsing/window-classify": [
    { label: "Call Parsing", to: "/app/call-parsing" },
    { label: "Window Classify" },
  ],
  "/app/sequence-models/continuous-embedding": [
    { label: "Sequence Models", to: "/app/sequence-models" },
    { label: "Continuous Embedding" },
  ],
  "/app/sequence-models/hmm-sequence": [
    { label: "Sequence Models", to: "/app/sequence-models" },
    { label: "HMM Sequence" },
  ],
  "/app/sequence-models/masked-transformer": [
    { label: "Sequence Models", to: "/app/sequence-models" },
    { label: "Masked Transformer" },
  ],
  "/app/admin": [{ label: "Admin" }],
};

export function Breadcrumbs() {
  const { pathname } = useLocation();
  const { jobId } = useParams<{ jobId?: string }>();

  let crumbs: Crumb[];

  if (
    jobId &&
    pathname.startsWith("/app/sequence-models/continuous-embedding/")
  ) {
    crumbs = [
      { label: "Sequence Models", to: "/app/sequence-models" },
      {
        label: "Continuous Embedding",
        to: "/app/sequence-models/continuous-embedding",
      },
      { label: `Job ${jobId.slice(0, 8)}` },
    ];
  } else if (
    jobId &&
    pathname.startsWith("/app/sequence-models/hmm-sequence/")
  ) {
    crumbs = [
      { label: "Sequence Models", to: "/app/sequence-models" },
      { label: "HMM Sequence", to: "/app/sequence-models/hmm-sequence" },
      { label: `Job ${jobId.slice(0, 8)}` },
    ];
  } else if (
    jobId &&
    pathname.startsWith("/app/sequence-models/masked-transformer/")
  ) {
    crumbs = [
      { label: "Sequence Models", to: "/app/sequence-models" },
      {
        label: "Masked Transformer",
        to: "/app/sequence-models/masked-transformer",
      },
      { label: `Job ${jobId.slice(0, 8)}` },
    ];
  } else {
    crumbs = staticRoutes[pathname] ?? [
      { label: "Call Parsing", to: "/app/call-parsing" },
      { label: "Detection" },
    ];
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

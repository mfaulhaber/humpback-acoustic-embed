import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";

interface ComputeDeviceBadgeProps {
  device: string | null;
  fallbackReason: string | null;
  className?: string;
}

const GREEN = "bg-green-100 text-green-800 border-green-200";
const NEUTRAL = "bg-gray-100 text-gray-700 border-gray-200";
const YELLOW = "bg-yellow-100 text-yellow-900 border-yellow-300";

export function ComputeDeviceBadge({
  device,
  fallbackReason,
  className,
}: ComputeDeviceBadgeProps) {
  if (!device) return null;

  if (device === "mps") {
    return (
      <Badge variant="outline" className={cn(GREEN, className)}>
        MPS
      </Badge>
    );
  }

  if (device === "cuda") {
    return (
      <Badge variant="outline" className={cn(GREEN, className)}>
        CUDA
      </Badge>
    );
  }

  if (device === "cpu" && fallbackReason) {
    return (
      <Badge variant="outline" className={cn(YELLOW, className)}>
        CPU (fallback: {fallbackReason})
      </Badge>
    );
  }

  if (device === "cpu") {
    return (
      <Badge variant="outline" className={cn(NEUTRAL, className)}>
        CPU
      </Badge>
    );
  }

  return null;
}

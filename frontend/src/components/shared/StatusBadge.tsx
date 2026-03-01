import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";

const statusColors: Record<string, string> = {
  queued: "bg-blue-100 text-blue-800 border-blue-200",
  running: "bg-yellow-100 text-yellow-800 border-yellow-200",
  complete: "bg-green-100 text-green-800 border-green-200",
  failed: "bg-red-100 text-red-800 border-red-200",
  canceled: "bg-gray-100 text-gray-600 border-gray-200",
};

interface StatusBadgeProps {
  status: string;
  className?: string;
}

export function StatusBadge({ status, className }: StatusBadgeProps) {
  return (
    <Badge variant="outline" className={cn(statusColors[status] ?? "bg-gray-100 text-gray-600", className)}>
      {status}
    </Badge>
  );
}

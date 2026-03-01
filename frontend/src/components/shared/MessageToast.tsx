import { toast } from "@/components/ui/use-toast";

export function showMsg(type: "success" | "error" | "warning", message: string) {
  toast({
    title: type.charAt(0).toUpperCase() + type.slice(1),
    description: message,
    variant: type === "error" ? "destructive" : "default",
    duration: 4000,
  });
}

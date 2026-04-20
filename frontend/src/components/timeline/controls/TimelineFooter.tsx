import React from "react";
import { COLORS } from "../constants";

interface TimelineFooterProps {
  children: React.ReactNode;
}

export function TimelineFooter({ children }: TimelineFooterProps) {
  return (
    <div
      className="flex flex-col gap-1 py-2 px-4"
      style={{
        background: COLORS.bg,
        borderTop: `1px solid ${COLORS.border}`,
      }}
    >
      {children}
    </div>
  );
}

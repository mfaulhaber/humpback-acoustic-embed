import { Link } from "react-router-dom";
import { Waves, Settings } from "lucide-react";

export function TopNav() {
  return (
    <header className="fixed top-0 left-0 right-0 h-12 bg-slate-800 text-white px-4 flex items-center justify-between z-50">
      <Link to="/app/audio" className="flex items-center gap-2 hover:opacity-90">
        <Waves className="h-5 w-5" />
        <span className="font-semibold text-sm">Humpback Acoustic</span>
      </Link>
      <Link
        to="/app/admin"
        className="p-1.5 rounded-md hover:bg-slate-700 transition-colors"
        aria-label="Settings"
      >
        <Settings className="h-4 w-4" />
      </Link>
    </header>
  );
}

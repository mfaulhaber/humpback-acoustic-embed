import { useState, useCallback, useEffect, useRef } from "react";

type ToastVariant = "default" | "destructive";

interface Toast {
  id: string;
  title?: string;
  description?: string;
  variant?: ToastVariant;
}

let listeners: Array<(toasts: Toast[]) => void> = [];
let memoryToasts: Toast[] = [];
let counter = 0;

function dispatch(toasts: Toast[]) {
  memoryToasts = toasts;
  listeners.forEach((l) => l(toasts));
}

export function toast(props: Omit<Toast, "id"> & { duration?: number }) {
  const id = String(counter++);
  const newToast: Toast = { ...props, id };
  dispatch([...memoryToasts, newToast]);

  const dur = props.duration ?? 4000;
  setTimeout(() => {
    dispatch(memoryToasts.filter((t) => t.id !== id));
  }, dur);

  return id;
}

export function useToast() {
  const [toasts, setToasts] = useState<Toast[]>(memoryToasts);
  const setToastsRef = useRef(setToasts);
  setToastsRef.current = setToasts;

  useEffect(() => {
    const listener = (t: Toast[]) => setToastsRef.current(t);
    listeners.push(listener);
    return () => {
      listeners = listeners.filter((l) => l !== listener);
    };
  }, []);

  const dismiss = useCallback((id: string) => {
    dispatch(memoryToasts.filter((t) => t.id !== id));
  }, []);

  return { toasts, toast, dismiss };
}

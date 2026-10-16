"use client";
import { ReactNode, useEffect, useRef, useState } from "react";

/**
 * Renders `children` only once the placeholder comes within `margin` of the viewport. Heavy
 * below-the-fold charts (thousands of SVG nodes) then don't cost main-thread time on load.
 * The placeholder reserves `minHeight` so nothing shifts when the content mounts.
 */
export function LazyMount({ children, minHeight, margin = "300px" }: { children: ReactNode; minHeight: number; margin?: string }) {
  const ref = useRef<HTMLDivElement>(null);
  const [visible, setVisible] = useState(false);

  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    if (typeof IntersectionObserver === "undefined") {
      const id = window.setTimeout(() => setVisible(true), 0); // no IO (old browsers, tests): just show it
      return () => window.clearTimeout(id);
    }
    const io = new IntersectionObserver(
      (entries) => {
        if (entries.some((e) => e.isIntersecting)) {
          setVisible(true);
          io.disconnect();
        }
      },
      { rootMargin: margin },
    );
    io.observe(el);
    return () => io.disconnect();
  }, [margin]);

  return (
    <div ref={ref} style={visible ? undefined : { minHeight }}>
      {visible ? children : null}
    </div>
  );
}

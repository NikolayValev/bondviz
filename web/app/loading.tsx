import { Skeleton } from "@/components/ui/Skeleton";

// Instant fallback while a route's code and data stream in. Sized like a typical tool page
// (title, controls row, main chart) so the swap doesn't shift the layout.
export default function Loading() {
  return (
    <div role="status" aria-label="Loading" className="space-y-6">
      <Skeleton className="h-9 w-64" />
      <Skeleton className="h-24" />
      <Skeleton className="h-80" />
    </div>
  );
}

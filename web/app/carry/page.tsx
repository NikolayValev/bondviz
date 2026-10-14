import { CarryClient } from "./CarryClient";
import { toolMetadata } from "@/lib/tools";

export const metadata = toolMetadata("/carry");

export default function CarryPage() {
  return <CarryClient />;
}

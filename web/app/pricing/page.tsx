import { PricingClient } from "./PricingClient";
import { toolMetadata } from "@/lib/tools";

export const metadata = toolMetadata("/pricing");

export default function PricingPage() {
  return <PricingClient />;
}

import { PcaClient } from "./PcaClient";
import { toolMetadata } from "@/lib/tools";

export const metadata = toolMetadata("/pca");

export default function PcaPage() {
  return <PcaClient />;
}

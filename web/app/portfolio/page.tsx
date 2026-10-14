import { PortfolioClient } from "./PortfolioClient";
import { toolMetadata } from "@/lib/tools";

export const metadata = toolMetadata("/portfolio");

export default function PortfolioPage() {
  return <PortfolioClient />;
}

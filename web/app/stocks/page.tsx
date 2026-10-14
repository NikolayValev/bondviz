import { StocksClient } from "./StocksClient";
import { toolMetadata } from "@/lib/tools";

export const metadata = toolMetadata("/stocks");

export default function StocksPage() {
  return <StocksClient />;
}

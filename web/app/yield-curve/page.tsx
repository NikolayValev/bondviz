import { YieldCurveClient } from "./YieldCurveClient";
import { toolMetadata } from "@/lib/tools";

export const metadata = toolMetadata("/yield-curve");

export default function YieldCurvePage() {
  return <YieldCurveClient />;
}

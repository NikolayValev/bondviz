import { SignalClient } from "./SignalClient";
import { toolMetadata } from "@/lib/tools";

export const metadata = toolMetadata("/signal");

export default function SignalPage() {
  return <SignalClient />;
}

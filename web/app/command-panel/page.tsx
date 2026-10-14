import { CommandPanelClient } from "./CommandPanelClient";
import { toolMetadata } from "@/lib/tools";

export const metadata = toolMetadata("/command-panel");

export default function CommandPanelPage() {
  return <CommandPanelClient />;
}

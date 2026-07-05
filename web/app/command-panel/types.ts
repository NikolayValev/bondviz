import type { PinnedWidget } from "@nikolayvalev/command-panel";

/** A dashboard widget sent to the chat to be improved (nonce re-triggers repeats). */
export interface ImproveRequest {
  widget: PinnedWidget;
  nonce: number;
}

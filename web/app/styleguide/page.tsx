import { StyleguideClient } from "./StyleguideClient";
import { pageMetadata } from "@/lib/site";

export const metadata = pageMetadata({
  title: "Design system",
  description: "Every BondViz design token, UI primitive and chart, rendered from the live components.",
  path: "/styleguide",
});

export default function StyleguidePage() {
  return <StyleguideClient />;
}

// Chat/generation route. Server-only: the AI SDK provider and the GitHub Models
// token never reach the client bundle (imports from the /server entry).

import { createCommandPanelHandler } from "@nikolayvalev/command-panel/server";
import { defaultComponentRegistry } from "@nikolayvalev/command-panel";
import { createOpenAI } from "@ai-sdk/openai";
import { dataRegistry } from "../../command-panel/data";

const github = createOpenAI({
  baseURL: "https://models.github.ai/inference",
  apiKey: process.env.GITHUB_MODELS_TOKEN,
});

export function POST(req: Request) {
  return createCommandPanelHandler({
    componentRegistry: defaultComponentRegistry,
    dataRegistry,
    model: github.chat(process.env.GITHUB_MODELS_MODEL ?? "openai/gpt-4o"),
  })(req);
}

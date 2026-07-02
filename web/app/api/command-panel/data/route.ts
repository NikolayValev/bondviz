// Read-only data route backing useMetric in generated widgets. Resolves only
// ids registered in the DataRegistry; unknown ids are rejected by the handler.

import { createDataRouteHandler } from "@nikolayvalev/command-panel";
import { dataRegistry } from "../../../command-panel/data";

export const POST = createDataRouteHandler(dataRegistry);

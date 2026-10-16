// Typed, compact query-string state. A page declares a schema (one codec per field) and its
// defaults; encodeState writes only the fields that differ from the defaults, and decodeState
// reads back whatever is present and valid, falling back to defaults field by field. A
// hand-edited or truncated URL therefore degrades to a sensible view instead of an error.

export interface Codec<T> {
  encode(value: T): string;
  /** undefined means "missing or invalid": the field keeps its default. */
  decode(raw: string): T | undefined;
}

export type Schema<S> = { [K in keyof S]: Codec<S[K]> };

const trimNum = (v: number) => String(+v.toFixed(6));

/** A finite number, optionally clamped to [min, max]. */
export function num({ min = -Infinity, max = Infinity }: { min?: number; max?: number } = {}): Codec<number> {
  return {
    encode: trimNum,
    decode: (raw) => {
      if (raw.trim() === "") return undefined;
      const n = Number(raw);
      return Number.isFinite(n) ? Math.min(max, Math.max(min, n)) : undefined;
    },
  };
}

/** One of a fixed set of values (strings or numbers). */
export function oneOf<T extends string | number>(values: readonly T[]): Codec<T> {
  return {
    encode: (v) => String(v),
    decode: (raw) => values.find((v) => String(v) === raw),
  };
}

/** Free text, capped in length. */
export function text(maxLength = 200): Codec<string> {
  return { encode: (v) => v, decode: (raw) => raw.slice(0, maxLength) };
}

/** A list of numbers drawn from an allowed set: "25,50,100". Unknown entries are dropped. */
export function numSet(allowed: readonly number[]): Codec<number[]> {
  return {
    encode: (v) => v.join(","),
    decode: (raw) => {
      if (raw === "") return [];
      const out = raw.split(",").map(Number).filter((n) => allowed.includes(n));
      return [...new Set(out)];
    },
  };
}

// encodeURIComponent leaves "~" alone, and "~" is the item separator, so escape it too.
const escapeField = (s: string) => encodeURIComponent(s).replace(/~/g, "%7E");

/**
 * A list of records, each encoded as comma-separated fields with items joined by "~".
 * Text fields are percent-encoded so labels may contain the separators. An item that fails
 * to decode is dropped; at most `maxItems` are read.
 */
export function records<T extends object>(
  fields: { [K in keyof T]: Codec<T[K]> },
  maxItems = 50,
): Codec<T[]> {
  const keys = Object.keys(fields) as (keyof T)[];
  return {
    encode: (items) =>
      items.map((item) => keys.map((k) => escapeField(fields[k].encode(item[k]))).join(",")).join("~"),
    decode: (raw) => {
      if (raw === "") return [];
      const out: T[] = [];
      for (const chunk of raw.split("~").slice(0, maxItems)) {
        const parts = chunk.split(",");
        if (parts.length !== keys.length) continue;
        const item = {} as T;
        let ok = true;
        keys.forEach((k, i) => {
          let piece: string;
          try {
            piece = decodeURIComponent(parts[i]);
          } catch {
            ok = false;
            return;
          }
          const v = fields[k].decode(piece);
          if (v === undefined) ok = false;
          else item[k] = v;
        });
        if (ok) out.push(item);
      }
      return out;
    },
  };
}

/** Query params for the fields that differ from their defaults (compared by encoding). */
export function encodeState<S>(schema: Schema<S>, state: S, defaults: S): URLSearchParams {
  const params = new URLSearchParams();
  for (const key of Object.keys(schema) as (keyof S & string)[]) {
    const codec = schema[key];
    const value = codec.encode(state[key]);
    if (value !== codec.encode(defaults[key])) params.set(key, value);
  }
  return params;
}

/** State from query params: each present, valid field overrides its default. */
export function decodeState<S>(schema: Schema<S>, params: URLSearchParams, defaults: S): S {
  const out = { ...defaults };
  for (const key of Object.keys(schema) as (keyof S & string)[]) {
    const raw = params.get(key);
    if (raw === null) continue;
    const value = schema[key].decode(raw);
    if (value !== undefined) out[key] = value;
  }
  return out;
}

/** Merge a page's state params into an existing query string, leaving unrelated params alone. */
export function mergeQuery<S>(schema: Schema<S>, current: string, state: S, defaults: S): string {
  const params = new URLSearchParams(current);
  for (const key of Object.keys(schema)) params.delete(key);
  encodeState(schema, state, defaults).forEach((v, k) => params.set(k, v));
  const qs = params.toString();
  return qs ? `?${qs}` : "";
}

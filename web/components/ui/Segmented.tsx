"use client";

export interface SegmentedOption<T extends string | number> {
  label: string;
  value: T;
}

/** A compact segmented control. Single-select by default; set multi for a
 *  toggle group (used for scenario shift magnitudes). */
export function Segmented<T extends string | number>({
  options,
  value,
  onChange,
  multi = false,
  ariaLabel,
}: {
  options: SegmentedOption<T>[];
  value: T[] | T;
  onChange: (next: T[] | T) => void;
  multi?: boolean;
  ariaLabel?: string;
}) {
  const selected = Array.isArray(value) ? value : [value];
  const isOn = (v: T) => selected.includes(v);

  const toggle = (v: T) => {
    if (!multi) return onChange(v);
    const set = new Set(selected);
    if (set.has(v)) set.delete(v);
    else set.add(v);
    onChange([...set]);
  };

  return (
    <div
      role={multi ? "group" : "radiogroup"}
      aria-label={ariaLabel}
      className="inline-flex flex-wrap gap-1 rounded-full border border-[var(--panel-border)] bg-[var(--panel-2)] p-1"
    >
      {options.map((o) => {
        const on = isOn(o.value);
        return (
          <button
            key={String(o.value)}
            type="button"
            role={multi ? "checkbox" : "radio"}
            aria-checked={on}
            onClick={() => toggle(o.value)}
            className={`tabnum rounded-full px-3 py-1.5 text-sm transition-all ${
              on
                ? "bg-[var(--accent)] font-semibold text-[var(--on-accent)] shadow-[0_0_0_1px_var(--accent-ring)]"
                : "text-[var(--muted)] hover:bg-white/5 hover:text-[var(--text)]"
            }`}
          >
            {o.label}
          </button>
        );
      })}
    </div>
  );
}

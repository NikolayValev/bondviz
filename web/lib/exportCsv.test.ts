import { describe, it, expect } from "vitest";
import { axisUnit, categoryTable, csvCell, exportFilename, seriesTable, toCsv } from "@/lib/exportCsv";

describe("csvCell", () => {
  it("writes numbers plainly to at most 6 decimals", () => {
    expect(csvCell(4.25)).toBe("4.25");
    expect(csvCell(0.1 + 0.2)).toBe("0.3");
    expect(csvCell(6.9244722928)).toBe("6.924472");
    expect(csvCell(1 / 12)).toBe("0.083333");
    expect(csvCell(-12.5)).toBe("-12.5");
    expect(csvCell(1_000_000)).toBe("1000000");
  });

  it("leaves missing and non-finite values empty", () => {
    expect(csvCell(null)).toBe("");
    expect(csvCell(undefined)).toBe("");
    expect(csvCell(NaN)).toBe("");
    expect(csvCell(Infinity)).toBe("");
  });

  it("quotes text holding separators, quotes or line breaks", () => {
    expect(csvCell("10Y")).toBe("10Y");
    expect(csvCell("Zero, annual")).toBe('"Zero, annual"');
    expect(csvCell('6" pipe')).toBe('"6"" pipe"');
    expect(csvCell("a\nb")).toBe('"a\nb"');
  });

  it("defuses text a spreadsheet would run as a formula", () => {
    expect(csvCell("=HYPERLINK(1)")).toBe("'=HYPERLINK(1)");
    expect(csvCell("+1")).toBe("'+1");
    expect(csvCell("-1")).toBe("'-1");
    expect(csvCell("@SUM(A1)")).toBe("'@SUM(A1)");
    expect(csvCell("=A1,B1")).toBe(`"'=A1,B1"`);
  });
});

describe("toCsv", () => {
  it("writes the header then each row, CRLF-terminated", () => {
    expect(toCsv({ columns: ["Tenor", "Yield (%)"], rows: [["2Y", 4.1], ["10Y", null]] })).toBe(
      "Tenor,Yield (%)\r\n2Y,4.1\r\n10Y,\r\n",
    );
  });
});

describe("seriesTable", () => {
  it("joins series on x, sorted, leaving gaps empty", () => {
    const t = seriesTable(
      [
        { label: "Par", points: [[2, 4.1], [0.5, 4.3]] },
        { label: "Zero", points: [[0.5, 4.35], [1, 4.2]] },
      ],
      { xLabel: "Maturity (years)" },
    );
    expect(t.columns).toEqual(["Maturity (years)", "Par", "Zero"]);
    expect(t.rows).toEqual([
      [0.5, 4.3, 4.35],
      [1, undefined, 4.2],
      [2, 4.1, undefined],
    ]);
  });

  it("adds the unit to each series header", () => {
    const t = seriesTable([{ label: "Zero", points: [[1, 4.2]] }], { xLabel: "Maturity (years)", unit: "%" });
    expect(t.columns).toEqual(["Maturity (years)", "Zero (%)"]);
  });

  it("writes time x values as ISO dates", () => {
    const day = Date.UTC(2026, 9, 20);
    const t = seriesTable([{ label: "2s10s", points: [[day, 12]] }], { xLabel: "Date", xType: "time" });
    expect(t.rows).toEqual([["2026-10-20", 12]]);
  });
});

describe("axisUnit", () => {
  it("reads the unit from an axis label's trailing parentheses", () => {
    expect(axisUnit("Rate (%)")).toBe("%");
    expect(axisUnit("Spread (bps)")).toBe("bps");
    expect(axisUnit("Maturity")).toBeUndefined();
    expect(axisUnit(undefined)).toBeUndefined();
  });
});

describe("categoryTable", () => {
  it("puts one category per row and one series per column", () => {
    const t = categoryTable("Tenor", ["2Y", "5Y"], [
      { label: "Carry", values: [10, null] },
      { label: "Roll", values: [3, 4] },
    ]);
    expect(t).toEqual({ columns: ["Tenor", "Carry", "Roll"], rows: [["2Y", 10, 3], ["5Y", null, 4]] });
  });
});

describe("exportFilename", () => {
  const date = new Date(Date.UTC(2026, 9, 27, 12));
  it("slugs the title and stamps the date", () => {
    expect(exportFilename("Zero & forward curve (bootstrapped)", "png", date)).toBe(
      "bondviz-zero-forward-curve-bootstrapped-2026-10-27.png",
    );
  });
  it("falls back when the title has no usable characters", () => {
    expect(exportFilename("—", "csv", date)).toBe("bondviz-export-2026-10-27.csv");
  });
});

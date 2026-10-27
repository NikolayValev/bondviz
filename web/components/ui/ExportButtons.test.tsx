import { describe, it, expect, vi, afterEach } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import { ExportButtons } from "@/components/ui/ExportButtons";

afterEach(() => vi.restoreAllMocks());

/** Capture what a click would download instead of navigating. */
function captureDownloads() {
  const blobs: Blob[] = [];
  const names: string[] = [];
  URL.createObjectURL = vi.fn((b: Blob) => {
    blobs.push(b);
    return "blob:test";
  });
  URL.revokeObjectURL = vi.fn();
  vi.spyOn(HTMLAnchorElement.prototype, "click").mockImplementation(function (this: HTMLAnchorElement) {
    names.push(this.download);
  });
  return { blobs, names };
}

describe("ExportButtons", () => {
  it("downloads the table as a UTF-8 CSV named after the title", async () => {
    const { blobs, names } = captureDownloads();
    render(
      <ExportButtons title="Carry and roll-down 3M" csv={() => ({ columns: ["Tenor", "Carry (bps)"], rows: [["2Y", 101.5]] })} />,
    );
    fireEvent.click(screen.getByRole("button", { name: "Download Carry and roll-down 3M data as CSV" }));
    expect(names[0]).toMatch(/^bondviz-carry-and-roll-down-3m-\d{4}-\d{2}-\d{2}\.csv$/);
    expect(blobs[0].type).toBe("text/csv;charset=utf-8");
    const bytes = new Uint8Array(await blobs[0].arrayBuffer());
    expect([...bytes.slice(0, 3)]).toEqual([0xef, 0xbb, 0xbf]); // BOM, so Excel reads UTF-8
    expect(new TextDecoder().decode(bytes.slice(3))).toBe("Tenor,Carry (bps)\r\n2Y,101.5\r\n");
  });

  it("shows only the buttons it has a source for", () => {
    render(<ExportButtons title="T" csv={() => ({ columns: [], rows: [] })} />);
    expect(screen.getByRole("button", { name: /as CSV/ })).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /as PNG/ })).toBeNull();
  });

  it("says so when the chart can't be captured", async () => {
    render(<ExportButtons title="T" svg={() => null} />);
    fireEvent.click(screen.getByRole("button", { name: "Download T chart as PNG" }));
    expect(await screen.findByRole("status")).toHaveTextContent("Image export failed");
  });
});

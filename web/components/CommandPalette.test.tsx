import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, fireEvent, act } from "@testing-library/react";
import { CommandPalette, OPEN_PALETTE_EVENT } from "@/components/CommandPalette";

const push = vi.fn();
vi.mock("next/navigation", () => ({ useRouter: () => ({ push }) }));

beforeEach(() => push.mockReset());

const openWithShortcut = () => fireEvent.keyDown(window, { key: "k", ctrlKey: true });

describe("CommandPalette", () => {
  it("opens on Ctrl-K, focuses the search, and lists every destination", () => {
    render(<CommandPalette />);
    expect(screen.queryByRole("dialog")).toBeNull();
    openWithShortcut();
    expect(screen.getByRole("dialog", { name: "Go to a page" })).toBeInTheDocument();
    expect(screen.getByRole("combobox")).toHaveFocus();
    expect(screen.getAllByRole("option").length).toBeGreaterThanOrEqual(11);
  });

  it("filters, moves with the arrow keys, and navigates on Enter", () => {
    render(<CommandPalette />);
    openWithShortcut();
    const input = screen.getByRole("combobox");
    fireEvent.change(input, { target: { value: "yield" } });
    const options = screen.getAllByRole("option");
    expect(options.map((o) => o.textContent)).toEqual([expect.stringContaining("Yield Curve Explorer"), expect.stringContaining("Yield Curve PCA")]);
    expect(options[0]).toHaveAttribute("aria-selected", "true");
    fireEvent.keyDown(input, { key: "ArrowDown" });
    expect(input).toHaveAttribute("aria-activedescendant", options[1].id);
    fireEvent.keyDown(input, { key: "Enter" });
    expect(push).toHaveBeenCalledWith("/pca");
    expect(screen.queryByRole("dialog")).toBeNull();
  });

  it("closes on Escape and returns focus to where it was", () => {
    render(
      <>
        <button>before</button>
        <CommandPalette />
      </>,
    );
    const before = screen.getByRole("button", { name: "before" });
    before.focus();
    openWithShortcut();
    fireEvent.keyDown(screen.getByRole("combobox"), { key: "Escape" });
    expect(screen.queryByRole("dialog")).toBeNull();
    expect(before).toHaveFocus();
  });

  it("opens from the nav button's event and shows an empty state", () => {
    render(<CommandPalette />);
    act(() => {
      window.dispatchEvent(new Event(OPEN_PALETTE_EVENT));
    });
    fireEvent.change(screen.getByRole("combobox"), { target: { value: "zzzz" } });
    expect(screen.getByText("No matching page.")).toBeInTheDocument();
  });
});

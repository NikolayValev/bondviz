import { describe, it, expect, vi } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import NotFound from "@/app/not-found";
import ErrorBoundary from "@/app/error";
import { TOOLS } from "@/lib/tools";

describe("not-found", () => {
  it("links home and to every tool", () => {
    render(<NotFound />);
    expect(screen.getByRole("link", { name: "Back to home" })).toHaveAttribute("href", "/");
    for (const t of TOOLS) expect(screen.getByRole("link", { name: t.title })).toHaveAttribute("href", t.href);
  });
});

describe("error boundary", () => {
  it("offers a retry that calls unstable_retry and shows the digest", () => {
    const retry = vi.fn();
    vi.spyOn(console, "error").mockImplementation(() => {});
    render(<ErrorBoundary error={Object.assign(new Error("boom"), { digest: "abc123" })} unstable_retry={retry} />);
    expect(screen.getByRole("alert")).toHaveTextContent("Something went wrong");
    expect(screen.getByText(/abc123/)).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Try again" }));
    expect(retry).toHaveBeenCalledOnce();
  });
});

import { describe, it, expect } from "vitest";
import { render } from "@testing-library/react";
import { CategoryBarChart } from "@/components/charts/CategoryBarChart";

describe("CategoryBarChart", () => {
  it("renders an svg with one rect per (series, category) cell", () => {
    const { container, getByRole } = render(
      <CategoryBarChart
        width={600}
        ariaLabel="test bars"
        categories={["2Y", "5Y", "10Y"]}
        series={[
          { id: "carry", label: "Carry", color: "#00d68f", values: [10, 20, 30] },
          { id: "roll", label: "Roll", color: "#5b8def", values: [5, -5, 15] },
        ]}
      />,
    );
    expect(getByRole("img", { name: "test bars" })).toBeTruthy();
    // 2 series × 3 categories = 6 bar rects (class "bar")
    expect(container.querySelectorAll("rect.bar").length).toBe(6);
  });
});

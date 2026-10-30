import { describe, it, expect, afterEach } from "vitest";
import { dayLabel, localIsoDate, monthYear, shiftMonths } from "@/lib/dates";

describe("shiftMonths", () => {
  it("moves by calendar months across year ends", () => {
    expect(shiftMonths("2026-10-13", -1)).toBe("2026-09-13");
    expect(shiftMonths("2026-01-15", -1)).toBe("2025-12-15");
    expect(shiftMonths("2026-10-13", -12)).toBe("2025-10-13");
    expect(shiftMonths("2025-10-16", 24)).toBe("2027-10-16");
  });

  it("clamps to the last day of a shorter month", () => {
    expect(shiftMonths("2026-03-31", -1)).toBe("2026-02-28");
    expect(shiftMonths("2024-03-30", -1)).toBe("2024-02-29"); // leap year
    expect(shiftMonths("2026-05-31", 1)).toBe("2026-06-30");
  });
});

describe("labels", () => {
  it("formats months and days in UTC", () => {
    expect(monthYear("2001-03-01")).toBe("Mar 2001");
    expect(dayLabel("2026-10-14")).toBe("Oct 14, 2026");
    expect(dayLabel(Date.UTC(2007, 3, 17))).toBe("Apr 17, 2007");
  });
});

describe("localIsoDate", () => {
  const tz = process.env.TZ;
  afterEach(() => {
    process.env.TZ = tz;
  });

  it("uses the viewer's calendar date, not UTC's", () => {
    process.env.TZ = "America/New_York";
    // 9pm on Oct 27 in New York is already Oct 28 in UTC.
    expect(localIsoDate(new Date("2026-10-28T01:00:00Z"))).toBe("2026-10-27");
  });
});

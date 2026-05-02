import { describe, expect, it, vi } from "vitest";
import { fireEvent, render, screen } from "@testing-library/react";
import { RegionNavBar, type RegionInfo } from "./RegionNavBar";

const regions: RegionInfo[] = [
  { regionId: "abcdef1234", startTimestamp: 0, endTimestamp: 10 },
  { regionId: "ghijkl5678", startTimestamp: 10, endTimestamp: 20 },
  { regionId: "mnopqr9012", startTimestamp: 20, endTimestamp: 30 },
];

describe("RegionNavBar", () => {
  it("displays current region count and id prefix", () => {
    render(
      <RegionNavBar
        regions={regions}
        activeIndex={1}
        onPrev={() => {}}
        onNext={() => {}}
        enableKeyboardShortcuts={false}
      />,
    );
    const label = screen.getByTestId("region-nav-bar");
    expect(label.textContent ?? "").toContain("Region 2/3");
    expect(label.textContent ?? "").toContain("ghijkl56");
  });

  it("triggers onPrev/onNext via A/D shortcuts", () => {
    const onPrev = vi.fn();
    const onNext = vi.fn();
    render(
      <RegionNavBar
        regions={regions}
        activeIndex={1}
        onPrev={onPrev}
        onNext={onNext}
      />,
    );
    fireEvent.keyDown(window, { key: "a" });
    fireEvent.keyDown(window, { key: "d" });
    expect(onPrev).toHaveBeenCalledTimes(1);
    expect(onNext).toHaveBeenCalledTimes(1);
  });

  it("disables prev at first index, next at last index", () => {
    const onPrev = vi.fn();
    const onNext = vi.fn();
    const { rerender } = render(
      <RegionNavBar
        regions={regions}
        activeIndex={0}
        onPrev={onPrev}
        onNext={onNext}
        enableKeyboardShortcuts={false}
      />,
    );
    expect(
      (screen.getByTestId("region-nav-prev") as HTMLButtonElement).disabled,
    ).toBe(true);
    expect(
      (screen.getByTestId("region-nav-next") as HTMLButtonElement).disabled,
    ).toBe(false);

    rerender(
      <RegionNavBar
        regions={regions}
        activeIndex={2}
        onPrev={onPrev}
        onNext={onNext}
        enableKeyboardShortcuts={false}
      />,
    );
    expect(
      (screen.getByTestId("region-nav-prev") as HTMLButtonElement).disabled,
    ).toBe(false);
    expect(
      (screen.getByTestId("region-nav-next") as HTMLButtonElement).disabled,
    ).toBe(true);
  });

  it("renders nothing when no regions", () => {
    const { container } = render(
      <RegionNavBar
        regions={[]}
        activeIndex={0}
        onPrev={() => {}}
        onNext={() => {}}
        enableKeyboardShortcuts={false}
      />,
    );
    expect(container.firstChild).toBeNull();
  });
});

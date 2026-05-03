import { describe, expect, it, vi } from "vitest";
import { fireEvent, render, screen } from "@testing-library/react";
import { MotifTokenCountSelector } from "./MotifTokenCountSelector";

describe("MotifTokenCountSelector", () => {
  it("clicking an inactive value fires onChange with that value", () => {
    const onChange = vi.fn();
    render(
      <MotifTokenCountSelector
        value={null}
        onChange={onChange}
        availableLengths={new Set([2, 3, 4])}
        isMotifsLoading={false}
      />,
    );
    fireEvent.click(screen.getByTestId("motif-token-count-3"));
    expect(onChange).toHaveBeenCalledWith(3);
  });

  it("clicking the active value fires onChange(null)", () => {
    const onChange = vi.fn();
    render(
      <MotifTokenCountSelector
        value={3}
        onChange={onChange}
        availableLengths={new Set([2, 3, 4])}
        isMotifsLoading={false}
      />,
    );
    fireEvent.click(screen.getByTestId("motif-token-count-3"));
    expect(onChange).toHaveBeenCalledWith(null);
  });

  it("disables buttons whose length is not in availableLengths", () => {
    render(
      <MotifTokenCountSelector
        value={null}
        onChange={() => {}}
        availableLengths={new Set([2])}
        isMotifsLoading={false}
      />,
    );
    expect(
      (screen.getByTestId("motif-token-count-2") as HTMLButtonElement).disabled,
    ).toBe(false);
    expect(
      (screen.getByTestId("motif-token-count-3") as HTMLButtonElement).disabled,
    ).toBe(true);
    expect(
      (screen.getByTestId("motif-token-count-4") as HTMLButtonElement).disabled,
    ).toBe(true);
    expect(
      screen.getByTestId("motif-token-count-3").getAttribute("title"),
    ).toBe("No length-3 motifs");
  });

  it("disables every button while motifs are loading and shows a spinner", () => {
    render(
      <MotifTokenCountSelector
        value={null}
        onChange={() => {}}
        availableLengths={new Set([2, 3, 4])}
        isMotifsLoading
      />,
    );
    for (const n of [2, 3, 4]) {
      expect(
        (screen.getByTestId(`motif-token-count-${n}`) as HTMLButtonElement)
          .disabled,
      ).toBe(true);
    }
    expect(screen.getByTestId("motif-token-count-spinner")).toBeTruthy();
  });

  it("aria-pressed reflects the active value", () => {
    render(
      <MotifTokenCountSelector
        value={4}
        onChange={() => {}}
        availableLengths={new Set([2, 3, 4])}
        isMotifsLoading={false}
      />,
    );
    expect(
      screen.getByTestId("motif-token-count-2").getAttribute("aria-pressed"),
    ).toBe("false");
    expect(
      screen.getByTestId("motif-token-count-4").getAttribute("aria-pressed"),
    ).toBe("true");
  });
});

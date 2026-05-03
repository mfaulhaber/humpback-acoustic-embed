import { describe, expect, it, vi } from "vitest";
import { fireEvent, render, screen } from "@testing-library/react";
import { LABEL_COLORS, labelColor } from "./constants";
import { MotifTimelineLegend } from "./MotifTimelineLegend";

describe("MotifTimelineLegend", () => {
  it("returns null when there is nothing to render (no motif, no nav, no slot)", () => {
    const { container } = render(
      <MotifTimelineLegend
        selectedMotifKey={null}
        selectedStates={[]}
        numLabels={4}
        occurrencesTotal={0}
        activeOccurrenceIndex={0}
        onPrev={() => {}}
        onNext={() => {}}
      />,
    );
    expect(container.firstChild).toBeNull();
  });

  it("renders the tokenSelector slot when supplied with no motif/nav (HMM-style consumers see no slot)", () => {
    render(
      <MotifTimelineLegend
        selectedMotifKey={null}
        selectedStates={[]}
        numLabels={4}
        occurrencesTotal={0}
        activeOccurrenceIndex={0}
        onPrev={() => {}}
        onNext={() => {}}
        tokenSelector={<span data-testid="slot-content">slot</span>}
      />,
    );
    expect(screen.getByTestId("motif-timeline-legend-token-selector")).toBeTruthy();
    expect(screen.getByTestId("slot-content").textContent).toBe("slot");
  });

  it("renders nav over occurrences without a single-motif selection (byLength mode)", () => {
    render(
      <MotifTimelineLegend
        selectedMotifKey={null}
        selectedStates={[]}
        numLabels={4}
        occurrencesTotal={5}
        activeOccurrenceIndex={2}
        onPrev={() => {}}
        onNext={() => {}}
      />,
    );
    expect(
      screen.getByTestId("motif-timeline-legend-counter").textContent,
    ).toBe("3 / 5");
    // No selected-motif swatches in byLength mode.
    expect(screen.queryByTestId("motif-timeline-legend-swatch-0")).toBeNull();
  });

  it("renders one swatch per state with palette-derived background colors", () => {
    render(
      <MotifTimelineLegend
        selectedMotifKey="23-50"
        selectedStates={[23, 50]}
        numLabels={100}
        occurrencesTotal={3}
        activeOccurrenceIndex={1}
        onPrev={() => {}}
        onNext={() => {}}
      />,
    );
    const a = screen.getByTestId("motif-timeline-legend-swatch-0") as HTMLElement;
    const b = screen.getByTestId("motif-timeline-legend-swatch-1") as HTMLElement;
    expect(a.style.backgroundColor).toBe(toRgb(labelColor(23, 100)));
    expect(b.style.backgroundColor).toBe(toRgb(labelColor(50, 100)));
  });

  it("counter text matches {idx+1} / {total}", () => {
    render(
      <MotifTimelineLegend
        selectedMotifKey="1-2"
        selectedStates={[1, 2]}
        numLabels={4}
        occurrencesTotal={7}
        activeOccurrenceIndex={3}
        onPrev={() => {}}
        onNext={() => {}}
      />,
    );
    expect(
      screen.getByTestId("motif-timeline-legend-counter").textContent,
    ).toBe("4 / 7");
  });

  it("disables prev at index 0 and next at total-1", () => {
    const { rerender } = render(
      <MotifTimelineLegend
        selectedMotifKey="x"
        selectedStates={[0]}
        numLabels={4}
        occurrencesTotal={3}
        activeOccurrenceIndex={0}
        onPrev={() => {}}
        onNext={() => {}}
      />,
    );
    expect(
      (screen.getByTestId("motif-timeline-legend-prev") as HTMLButtonElement)
        .disabled,
    ).toBe(true);
    expect(
      (screen.getByTestId("motif-timeline-legend-next") as HTMLButtonElement)
        .disabled,
    ).toBe(false);

    rerender(
      <MotifTimelineLegend
        selectedMotifKey="x"
        selectedStates={[0]}
        numLabels={4}
        occurrencesTotal={3}
        activeOccurrenceIndex={2}
        onPrev={() => {}}
        onNext={() => {}}
      />,
    );
    expect(
      (screen.getByTestId("motif-timeline-legend-prev") as HTMLButtonElement)
        .disabled,
    ).toBe(false);
    expect(
      (screen.getByTestId("motif-timeline-legend-next") as HTMLButtonElement)
        .disabled,
    ).toBe(true);
  });

  it("invokes onPrev / onNext when enabled", () => {
    const onPrev = vi.fn();
    const onNext = vi.fn();
    render(
      <MotifTimelineLegend
        selectedMotifKey="x"
        selectedStates={[0]}
        numLabels={4}
        occurrencesTotal={3}
        activeOccurrenceIndex={1}
        onPrev={onPrev}
        onNext={onNext}
      />,
    );
    fireEvent.click(screen.getByTestId("motif-timeline-legend-prev"));
    fireEvent.click(screen.getByTestId("motif-timeline-legend-next"));
    expect(onPrev).toHaveBeenCalledTimes(1);
    expect(onNext).toHaveBeenCalledTimes(1);
  });

  it("uses LABEL_COLORS palette for small numLabels", () => {
    render(
      <MotifTimelineLegend
        selectedMotifKey="2-5"
        selectedStates={[2, 5]}
        numLabels={10}
        occurrencesTotal={1}
        activeOccurrenceIndex={0}
        onPrev={() => {}}
        onNext={() => {}}
      />,
    );
    const a = screen.getByTestId("motif-timeline-legend-swatch-0") as HTMLElement;
    expect(a.style.backgroundColor).toBe(toRgb(LABEL_COLORS[2]));
  });
});

// jsdom serializes background-color to ``rgb(r, g, b)`` regardless of the
// hex / hsl value the component sets, so test expectations need to be in
// the same form. This helper accepts hex (#rrggbb) or hsl(...) input.
function toRgb(value: string): string {
  if (value.startsWith("#")) {
    const hex = value.slice(1);
    const r = parseInt(hex.slice(0, 2), 16);
    const g = parseInt(hex.slice(2, 4), 16);
    const b = parseInt(hex.slice(4, 6), 16);
    return `rgb(${r}, ${g}, ${b})`;
  }
  if (value.startsWith("hsl")) {
    const [h, s, l] = value
      .replace("hsl(", "")
      .replace(")", "")
      .split(",")
      .map((part) => part.trim());
    const sn = parseFloat(s) / 100;
    const ln = parseFloat(l) / 100;
    const c = (1 - Math.abs(2 * ln - 1)) * sn;
    const hp = parseFloat(h) / 60;
    const x = c * (1 - Math.abs((hp % 2) - 1));
    const m = ln - c / 2;
    let rp = 0, gp = 0, bp = 0;
    if (0 <= hp && hp < 1) [rp, gp, bp] = [c, x, 0];
    else if (hp < 2) [rp, gp, bp] = [x, c, 0];
    else if (hp < 3) [rp, gp, bp] = [0, c, x];
    else if (hp < 4) [rp, gp, bp] = [0, x, c];
    else if (hp < 5) [rp, gp, bp] = [x, 0, c];
    else [rp, gp, bp] = [c, 0, x];
    const r = Math.round((rp + m) * 255);
    const g = Math.round((gp + m) * 255);
    const b = Math.round((bp + m) * 255);
    return `rgb(${r}, ${g}, ${b})`;
  }
  return value;
}

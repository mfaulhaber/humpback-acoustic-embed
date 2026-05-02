import { describe, expect, it, vi } from "vitest";
import { fireEvent, render, screen } from "@testing-library/react";
import { MemoryRouter, Routes, Route, useSearchParams } from "react-router-dom";
import { KPicker, useSelectedK } from "./KPicker";

function HarnessReadback({ kValues }: { kValues: number[] }) {
  const k = useSelectedK(kValues);
  return <div data-testid="selected-k">{String(k)}</div>;
}

function ParamsReadback() {
  const [params] = useSearchParams();
  return <div data-testid="params-k">{params.get("k") ?? "none"}</div>;
}

describe("KPicker", () => {
  it("defaults to first k value when no URL param is set", () => {
    render(
      <MemoryRouter initialEntries={["/"]}>
        <KPicker kValues={[50, 100, 200]} />
        <HarnessReadback kValues={[50, 100, 200]} />
      </MemoryRouter>,
    );
    expect(screen.getByTestId("selected-k").textContent).toBe("50");
    expect(screen.getByTestId("k-picker-tab-50").getAttribute("aria-selected")).toBe("true");
  });

  it("reads existing ?k= URL param", () => {
    render(
      <MemoryRouter initialEntries={["/?k=100"]}>
        <KPicker kValues={[50, 100, 200]} />
        <HarnessReadback kValues={[50, 100, 200]} />
      </MemoryRouter>,
    );
    expect(screen.getByTestId("selected-k").textContent).toBe("100");
  });

  it("falls back to first k when URL param is not in k_values", () => {
    render(
      <MemoryRouter initialEntries={["/?k=999"]}>
        <KPicker kValues={[50, 100, 200]} />
        <HarnessReadback kValues={[50, 100, 200]} />
      </MemoryRouter>,
    );
    expect(screen.getByTestId("selected-k").textContent).toBe("50");
  });

  it("syncs URL param on click", async () => {
    render(
      <MemoryRouter initialEntries={["/"]}>
        <Routes>
          <Route
            path="/"
            element={
              <>
                <KPicker kValues={[50, 100, 200]} />
                <ParamsReadback />
              </>
            }
          />
        </Routes>
      </MemoryRouter>,
    );
    expect(screen.getByTestId("params-k").textContent).toBe("none");
    fireEvent.click(screen.getByTestId("k-picker-tab-200"));
    expect(screen.getByTestId("params-k").textContent).toBe("200");
  });

  it("renders an empty placeholder when no k_values configured", () => {
    render(
      <MemoryRouter>
        <KPicker kValues={[]} />
      </MemoryRouter>,
    );
    expect(screen.getByTestId("k-picker").textContent).toContain(
      "No k-values configured",
    );
  });
});

describe("useSelectedK", () => {
  it("returns null for empty k_values regardless of URL param", () => {
    render(
      <MemoryRouter initialEntries={["/?k=100"]}>
        <HarnessReadback kValues={[]} />
      </MemoryRouter>,
    );
    expect(screen.getByTestId("selected-k").textContent).toBe("null");
  });

  it("respects a custom paramName", () => {
    function Custom({ kValues }: { kValues: number[] }) {
      const k = useSelectedK(kValues, "tok");
      return <div data-testid="selected-k">{String(k)}</div>;
    }
    render(
      <MemoryRouter initialEntries={["/?tok=200&k=999"]}>
        <Custom kValues={[50, 100, 200]} />
      </MemoryRouter>,
    );
    expect(screen.getByTestId("selected-k").textContent).toBe("200");
    // Silence unused-import lint
    void vi.fn();
  });
});

import { afterEach, beforeAll, beforeEach, describe, expect, it } from "vitest";
import { fireEvent, render, screen } from "@testing-library/react";
import { CollapsiblePanelCard } from "./CollapsiblePanelCard";

const KEY = "test-panel";
const STORAGE_KEY = `seq-models:panel:${KEY}`;

function installLocalStorageMock(): Storage {
  const store = new Map<string, string>();
  const mock: Storage = {
    get length() {
      return store.size;
    },
    clear() {
      store.clear();
    },
    getItem(key: string) {
      return store.has(key) ? (store.get(key) as string) : null;
    },
    key(index: number) {
      return Array.from(store.keys())[index] ?? null;
    },
    removeItem(key: string) {
      store.delete(key);
    },
    setItem(key: string, value: string) {
      store.set(key, String(value));
    },
  };
  Object.defineProperty(window, "localStorage", {
    configurable: true,
    value: mock,
  });
  return mock;
}

describe("CollapsiblePanelCard", () => {
  beforeAll(() => {
    if (typeof window.localStorage?.setItem !== "function") {
      installLocalStorageMock();
    }
  });
  beforeEach(() => {
    window.localStorage.clear();
  });
  afterEach(() => {
    window.localStorage.clear();
  });

  it("renders open by default with title and children visible", () => {
    render(
      <CollapsiblePanelCard title="Hello" storageKey={KEY}>
        <div data-testid="child">payload</div>
      </CollapsiblePanelCard>,
    );
    expect(screen.getByText("Hello")).toBeTruthy();
    expect(screen.getByTestId("child")).toBeTruthy();
    expect(
      screen.getByTestId("collapsible-panel-toggle").getAttribute("aria-expanded"),
    ).toBe("true");
  });

  it("toggling unmounts children and updates localStorage", () => {
    render(
      <CollapsiblePanelCard title="Hello" storageKey={KEY}>
        <div data-testid="child">payload</div>
      </CollapsiblePanelCard>,
    );
    fireEvent.click(screen.getByTestId("collapsible-panel-toggle"));
    expect(screen.queryByTestId("child")).toBeNull();
    expect(window.localStorage.getItem(STORAGE_KEY)).toBe("false");
  });

  it("starts closed when localStorage is pre-populated to false", () => {
    window.localStorage.setItem(STORAGE_KEY, "false");
    render(
      <CollapsiblePanelCard title="Hello" storageKey={KEY}>
        <div data-testid="child">payload</div>
      </CollapsiblePanelCard>,
    );
    expect(screen.queryByTestId("child")).toBeNull();
  });

  it("starts closed when defaultOpen=false and no localStorage entry", () => {
    render(
      <CollapsiblePanelCard title="Hello" storageKey={KEY} defaultOpen={false}>
        <div data-testid="child">payload</div>
      </CollapsiblePanelCard>,
    );
    expect(screen.queryByTestId("child")).toBeNull();
  });

  it("renders headerExtra and clicking inside it does not toggle the panel", () => {
    render(
      <CollapsiblePanelCard
        title="Hello"
        storageKey={KEY}
        headerExtra={
          <button type="button" data-testid="extra-btn">
            extra
          </button>
        }
      >
        <div data-testid="child">payload</div>
      </CollapsiblePanelCard>,
    );
    expect(screen.getByTestId("extra-btn")).toBeTruthy();
    fireEvent.click(screen.getByTestId("extra-btn"));
    expect(screen.getByTestId("child")).toBeTruthy();
    expect(window.localStorage.getItem(STORAGE_KEY)).toBeNull();
  });

  it("uses provided testId for root and toggle", () => {
    render(
      <CollapsiblePanelCard title="Hello" storageKey={KEY} testId="my-panel">
        <div>x</div>
      </CollapsiblePanelCard>,
    );
    expect(screen.getByTestId("my-panel")).toBeTruthy();
    expect(screen.getByTestId("my-panel-toggle")).toBeTruthy();
  });
});

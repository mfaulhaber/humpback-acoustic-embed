# Frontend Shell Domain

Load this domain for app shell navigation, breadcrumbs, shared UI components,
shared query hooks, admin UI, API client plumbing, or frontend-only state shared
across product domains.

## Primary Paths

- `frontend/src/App.tsx`
- `frontend/src/api/`
- `frontend/src/hooks/queries/`
- `frontend/src/components/layout/`
- `frontend/src/components/shared/`
- `frontend/src/components/ui/`
- `frontend/src/components/admin/`
- `frontend/e2e/navigation-retired-workflows.spec.ts`
- `frontend/e2e/compute-device-badge.spec.ts`

## Artifact Roots

- Built SPA output is served from `src/humpback/static/dist/` after production
  build.

## Likely Neighbors

- Load the owning feature domain for any route, query hook, or component that
  represents domain behavior.
- Load `signal-timeline` for shared timeline primitives and playback state.

## Before Editing

1. Identify whether the change is shell-only or changes domain behavior.
2. If domain behavior changes, load that domain capsule before editing.

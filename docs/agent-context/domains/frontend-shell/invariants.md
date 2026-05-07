# Frontend Shell Invariants

- Frontend package operations use `npm` from `frontend/`.
- The SPA uses React 18, Vite, TypeScript, Tailwind, shadcn/ui, and TanStack
  Query.
- Shared UI components should not encode domain-specific business rules.
- Route, navigation, and breadcrumb changes must stay consistent.
- Frontend operational time ranges should display UTC where project workflows
  use UTC.
- UI changes that affect data flow need TypeScript plus the owning domain's
  Playwright or API-level smoke tests.

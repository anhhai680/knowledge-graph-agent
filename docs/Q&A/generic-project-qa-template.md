# ASP.NET Core Web API + Clean Architecture — Project Q&A

Generated: 2025-08-11

Note: This template is database-agnostic. It fits .NET services using Clean Architecture (Domain, Application, Infrastructure, Presentation) regardless of persistence choice (EF Core, Dapper, Mongo, Cosmos, etc.). Replace examples with your domain.

## 1) What business capability does the service own and what are the core entities?

- Scope: Define the single bounded context this API owns (e.g., Listings, Orders, Catalog). Avoid mixed responsibilities.
- Core entities/aggregates: List primary models and invariants. Example: `Listing { Id, Title, Price, Status, SellerId, ... }`.
- Ownership and SLAs: Who are the consumers? Throughput/latency targets? Consistency expectations (strong/eventual)?
- Key decisions:
  - ID strategy (GUID vs ULID vs DB-generated vs natural key).
  - Multi-tenancy (per-tenant schema vs discriminator).
  - Deletion strategy (soft delete with `IsDeleted` vs hard delete).
  - Domain events needed? What invariants emit which events?

## 2) What are the API endpoints and expected behaviors?

- Resource routes: `/[entity]` with CRUD and task-specific commands.
  - GET `/{entity}`: list with pagination/filtering/sorting.
  - GET `/{entity}/{id}`: fetch by id.
  - POST `/{entity}`: create; return 201 with `Location` header.
  - PUT `/{entity}/{id}`: full replace; 204 on success.
  - PATCH `/{entity}/{id}`: partial update (JSON Patch or merge) — optional.
  - DELETE `/{entity}/{id}`: delete; 204 on success.
- Status codes: 200/201/204, 400 validation, 401/403 auth failures, 404 not found, 409 conflict, 422 domain rule violations.
- Pagination/Filtering: `?page=1&pageSize=20&sort=-createdAt&status=active` (pick and document a style).
- Errors: Consistent envelope using RFC 7807 Problem Details (`type`, `title`, `status`, `detail`, `instance`, plus `traceId`).
- Versioning: URL `/v1/...` or header-based; publish deprecation policy.
- Idempotency: Support `Idempotency-Key` on POST/PUT where clients may retry.

## 3) How is data modeled and persisted (database-agnostic)?

- Abstractions:
  - Define repositories/UoW interfaces in Application layer (e.g., `IListingRepository`, `IUnitOfWork`).
  - Infrastructure implements these interfaces using the chosen data access tech.
  - Domain models are persistence-ignorant (no ORM attributes in Domain).
- Modeling:
  - Value Objects vs Entities; aggregates with clear boundaries.
  - Concurrency control: optimistic via `Version` field or HTTP ETags.
  - Validation: DataAnnotations/FluentValidation for input; domain invariants in entities/services.
- Transactions & consistency:
  - Unit of Work for atomic operations; sagas/process managers for long-running or cross-aggregate workflows.
  - Outbox pattern for reliable event publishing.
- Configuration:
  - Connection strings and provider settings via `IOptions<T>` and environment variables.
  - Migrations/seeding strategy (e.g., EF Core migrations or custom scripts). Keep infra concerns in Infrastructure.

## 4) What’s the end-to-end workflow for create/update operations?

Example: Create [Entity]
- Validate request (schema + domain rules). Return 400/422 on failure.
- Check conflicts (uniques, related entity existence). Return 409 if violated.
- Map DTO -> Command -> Domain model; invoke application service/handler.
- Persist via repository/UoW; commit.
- Publish domain/integration event (e.g., `entity-created`) through a message bus; prefer Outbox.
- Return 201 with resource representation and `Location` header.

Example: Update [Entity]
- Retrieve current state; enforce concurrency (ETag/Version).
- Apply changes through domain methods; re-validate invariants.
- Persist and commit; return 204 or 200 depending on contract.
- Publish `entity-updated` event if applicable.

Contract (template)
- Input: JSON DTO with required/optional fields.
- Output: Resource representation or empty on 204; errors use Problem Details.
- Errors: 400/404/409/422 with actionable messages; include `traceId` for correlation.

## 5) What’s the architecture and what operational concerns apply?

- Layers (Clean Architecture):
  - Presentation (API Controllers/Minimal APIs)
  - Application (CQRS handlers/services, ports/interfaces, validators)
  - Domain (entities/value objects, domain events, business rules)
  - Infrastructure (DB provider, message bus, external services implementations)
- Cross-cutting:
  - DI registrations; pipeline behaviors (logging, validation, performance) — MediatR optional.
  - Mapping via AutoMapper or manual mappers.
  - Configuration via `IOptions<T>`; secrets from env/Key Vault.
- Security:
  - AuthN: JWT/OIDC; add `[Authorize]` and policies/roles.
  - Input hardening: model size limits, allow-lists for sorting/filtering fields.
- Observability:
  - Structured logging (Serilog), correlation IDs, OpenTelemetry traces/metrics/logs.
  - Health checks (`/health`), readiness/liveness, provider-specific checks in Infrastructure.
- API docs:
  - Swagger/OpenAPI with XML comments; enable in all envs (protect prod if needed).
- Performance:
  - Pagination defaults, selective projection, async I/O, connection pooling.
  - Caching: ETag/Cache-Control; server-side cache where safe.
- Delivery:
  - Local dev via Docker Compose (db/bus optional); seed data scripts.
  - CI/CD with build, test, security scans, infra migrations, and blue/green or canary where applicable.
- Testing:
  - Unit tests for Domain/Application; contract tests for API; integration tests with provider fakes or containers.

---

Quick adaptation checklist
- Define domain scope, entities, IDs, and invariants.
- Design API surface (routes, pagination, errors, versioning, idempotency).
- Choose persistence tech and implement repositories/UoW in Infrastructure.
- Add validation, auth, and observability; document with Swagger.
- Decide on domain/integration events and reliability (Outbox).
- Write unit/integration tests and add health checks.
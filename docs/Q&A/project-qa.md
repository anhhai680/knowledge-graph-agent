# Car Listing Service — Project Q&A

Generated: 2025-08-11

## 1) What business problem does this service solve and what’s the core entity?

- It provides a simple Car Listing service for a marketplace scenario.
- Core entity: `Car` with fields `Id`, `Brand`, `Model`, `Year`, `Mileage`, `Condition`, `Price`, and `Description`.
- Users can list all cars, view one by id, create a new listing, update, and delete.

## 2) What are the API endpoints and expected behaviors?

Base route: `/Car` (from `[Route("[controller]")]`).

- `GET /Car` — Returns all cars.
  - 200 OK with `Car[]`.
- `GET /Car/{id}` — Returns a specific car by id.
  - 200 OK with `Car` or 404 if not found.
- `POST /Car` — Creates a new car listing.
  - 201 Created with `Location: /Car/{id}` and created `Car` body.
- `PUT /Car/{id}` — Replaces an existing car by id.
  - 204 No Content or 404 if id doesn’t exist.
- `DELETE /Car/{id}` — Deletes a car by id.
  - 204 No Content or 404 if id doesn’t exist.

Swagger UI is enabled in Development environment.

## 3) How is data modeled and persisted?

- Persistence: MongoDB (connection string `ConnectionStrings:MongoDb` in `appsettings*.json`).
- Database: `CarMarketplace`; Collection: `Cars`.
- Model: `Car` uses MongoDB BSON attributes:
  - `[BsonId]` and `[BsonRepresentation(BsonType.ObjectId)]` for `Id` as an ObjectId stored as string.
- No explicit indexes or schema validation are defined in code; MongoDB is used schemaless.

## 4) What’s the end-to-end workflow when creating a car listing?

1. Client sends `POST /Car` with a `Car` payload (JSON).
2. Controller inserts the document with `InsertOneAsync` into MongoDB.
3. On success, returns `201 Created` with the created document and `Location` header to `GET /Car/{id}`.
4. A planned extension (TODO in code): publish a `car-listed` event to RabbitMQ after insertion.

## 5) What’s the overall architecture and notable operational characteristics?

- Architecture: ASP.NET Core Web API using Controllers; minimal hosting in `Program.cs`.
- Data access: Controller constructs `MongoClient` from `IConfiguration`, gets `CarMarketplace`/`Cars` collection, and performs CRUD directly.
- Security: Authorization middleware is added, but no auth schemes/policies are configured; endpoints are effectively anonymous.
- Observability: Default ASP.NET logging; Swagger/OpenAPI in Development.
- Constraints/Risks: No input validation or DTOs; no pagination or filtering; `PUT` is a full document replace; direct data access in controller (no service/repository layer); RabbitMQ integration is not implemented yet.
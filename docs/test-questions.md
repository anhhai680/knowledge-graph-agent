# Knowledge Graph Agent - Test Questions (Car Trading Platform)

Test questions focused on repository relationships, dependencies, and knowledge graph analysis for the car trading platform microservices: car-order-service, car-web-client, and car-notification-service.

## 1. Repository Relationship Analysis

### Q1: Repository dependency mapping
**Question:** "What are the dependencies and relationships between the three repositories in the car trading platform?"

**Expected JSON Output:**
```json
{
  "answer": "The repositories form a microservices ecosystem with these relationships:\n\n**Dependency Graph:**\n- car-web-client → car-order-service (HTTP REST API calls)\n- car-web-client → car-notification-service (HTTP REST API calls)\n- car-order-service → car-notification-service (RabbitMQ events)\n- car-order-service → car-listing-service (HTTP REST API calls - planned)\n\n**Shared Dependencies:**\n- All .NET services use .NET 8 Web API framework\n- RabbitMQ message broker for event-driven communication\n- MongoDB/PostgreSQL for data persistence\n- Swagger/OpenAPI for documentation",
  "context_snippets": [
    {
      "content": "## Tech Stack\n• .NET 8 Web API\n• PostgreSQL (or MySQL)\n• RabbitMQ (for event communication)",
      "file_path": "car-order-service/README.md",
      "line_range": [9, 11]
    },
    {
      "content": "## Tech Stack\n• .NET 8 Web API\n• RabbitMQ (or Kafka)\n• MongoDB or PostgreSQL (optional, for persistence)",
      "file_path": "car-notification-service/README.md",
      "line_range": [9, 11]
    }
  ],
  "source_files": ["car-order-service/README.md", "car-notification-service/README.md"],
  "citations": [
    {
      "file_path": "car-order-service/README.md",
      "line": 9,
      "snippet_id": "tech_dependencies"
    },
    {
      "file_path": "car-notification-service/README.md",
      "line": 9,
      "snippet_id": "notification_dependencies"
    }
  ],
  "confidence_score": 0.95,
  "follow_up_prompts": [
    "Show me the dependency graph in mermaid format",
    "What are the circular dependencies if any?",
    "How are version compatibility issues handled?"
  ]
}
```

### Q2: Cross-repository data flow analysis
**Question:** "Create a mermaid diagram showing the data flow and relationships between all repositories."

**Expected JSON Output:**
```json
{
  "answer": "Here's the complete data flow diagram across repositories:\n\n```mermaid\ngraph TB\n    subgraph \"Frontend Layer\"\n        WC[car-web-client<br/>React/TypeScript]\n    end\n    \n    subgraph \"Backend Services\"\n        OS[car-order-service<br/>.NET 8 API]\n        NS[car-notification-service<br/>.NET 8 API]\n        CLS[car-listing-service<br/>.NET 8 API]\n    end\n    \n    subgraph \"Data Layer\"\n        ODB[(PostgreSQL<br/>Orders)]\n        NDB[(MongoDB<br/>Notifications)]\n        CDB[(MongoDB<br/>Cars)]\n    end\n    \n    subgraph \"Infrastructure\"\n        RMQ[RabbitMQ<br/>Message Broker]\n        PG[Payment Gateway<br/>Mock]\n        EMAIL[Email/Push<br/>Providers]\n    end\n    \n    WC -.->|REST API| OS\n    WC -.->|REST API| NS\n    WC -.->|REST API| CLS\n    \n    OS -->|Events| RMQ\n    CLS -->|Events| RMQ\n    RMQ -->|Events| NS\n    \n    OS --> ODB\n    NS --> NDB\n    CLS --> CDB\n    \n    OS -.->|Payment| PG\n    NS -.->|Send| EMAIL\n    \n    OS -.->|Verify Cars| CLS\n```\n\nThis shows the complete ecosystem with REST APIs (dotted lines), event flows (solid lines), and data persistence relationships.",
  "context_snippets": [
    {
      "content": "// • Create order when a user clicks \"Buy Car\"\n// • Verify car status with car-listing-service (TODO)\n// • Communicate with payment gateway (mock)\n// • Update order status: pending, paid, cancelled\n// • Emits events: `order-created`, `order-paid` (TODO)",
      "file_path": "car-order-service/README.md",
      "line_range": [15, 19]
    }
  ],
  "source_files": ["car-order-service/README.md"],
  "citations": [
    {
      "file_path": "car-order-service/README.md",
      "line": 15,
      "snippet_id": "service_interactions"
    }
  ],
  "confidence_score": 0.93,
  "follow_up_prompts": [
    "How are data consistency issues handled across services?",
    "What happens when one service is down?",
    "Show me the event schema relationships"
  ]
}
```

## 2. Knowledge Graph Structure Analysis

### Q3: Entity relationship graph
**Question:** "What are the core entities and their relationships in the car trading knowledge graph?"

**Expected JSON Output:**
```json
{
  "answer": "Core entities and relationships in the car trading knowledge graph:\n\n```mermaid\nerDiagram\n    User {\n        string id PK\n        string email\n        string name\n        string role\n    }\n    \n    Car {\n        ObjectId id PK\n        string brand\n        string model\n        int year\n        decimal price\n        string sellerId FK\n        string status\n    }\n    \n    Order {\n        Guid id PK\n        string carId FK\n        string buyerId FK\n        string status\n        DateTime createdAt\n        decimal amount\n    }\n    \n    Notification {\n        Guid id PK\n        string userId FK\n        string entityType\n        string entityId\n        string message\n        bool isRead\n    }\n    \n    Repository {\n        string name PK\n        string technology\n        string database\n        string messageQueue\n    }\n    \n    User ||--o{ Car : \"sells\"\n    User ||--o{ Order : \"buys\"\n    Car ||--o{ Order : \"ordered\"\n    Order ||--o{ Notification : \"triggers\"\n    User ||--o{ Notification : \"receives\"\n    Repository ||--o{ User : \"manages\"\n    Repository ||--o{ Car : \"stores\"\n    Repository ||--o{ Order : \"processes\"\n    Repository ||--o{ Notification : \"sends\"\n```\n\nThis shows the complete knowledge graph with cross-repository entity relationships and data flows.",
  "context_snippets": [
    {
      "content": "public class Order\n{\n    public Guid Id { get; set; } = Guid.NewGuid();\n    public string CarId { get; set; } = string.Empty;\n    public string BuyerId { get; set; } = string.Empty;\n    public string Status { get; set; } = \"pending\";\n    public DateTime CreatedAt { get; set; } = DateTime.UtcNow;\n    public DateTime? PaidAt { get; set; };\n    public decimal Amount { get; set; };\n}",
      "file_path": "car-order-service/Order.cs",
      "line_range": [5, 14]
    }
  ],
  "source_files": ["car-order-service/Order.cs"],
  "citations": [
    {
      "file_path": "car-order-service/Order.cs",
      "line": 5,
      "snippet_id": "order_entity_graph"
    }
  ],
  "confidence_score": 0.94,
  "follow_up_prompts": [
    "How are entity relationships maintained across microservices?",
    "What graph traversal patterns are used?",
    "Show me the event propagation graph"
  ]
}
```

### Q4: Event-driven dependency graph
**Question:** "Create a mermaid graph showing the event dependencies and message flows between repositories."

**Expected JSON Output:**
```json
{
  "answer": "Event-driven dependency graph showing message flows:\n\n```mermaid\nflowchart TD\n    subgraph \"Event Publishers\"\n        CLS[car-listing-service]\n        OS[car-order-service]\n    end\n    \n    subgraph \"Event Types\"\n        E1[car-listed]\n        E2[car-sold]\n        E3[order-created]\n        E4[order-paid]\n        E5[order-cancelled]\n    end\n    \n    subgraph \"Message Infrastructure\"\n        RMQ[RabbitMQ Broker]\n        CQ[Car Events Queue]\n        OQ[Order Events Queue]\n        DLQ[Dead Letter Queue]\n    end\n    \n    subgraph \"Event Consumers\"\n        NS[car-notification-service]\n        WC[car-web-client via WebSocket]\n    end\n    \n    subgraph \"Event Handlers\"\n        CEH[Car Event Handler]\n        OEH[Order Event Handler]\n        EH[Email Handler]\n        PH[Push Handler]\n    end\n    \n    CLS --> E1\n    CLS --> E2\n    OS --> E3\n    OS --> E4\n    OS --> E5\n    \n    E1 --> RMQ\n    E2 --> RMQ\n    E3 --> RMQ\n    E4 --> RMQ\n    E5 --> RMQ\n    \n    RMQ --> CQ\n    RMQ --> OQ\n    RMQ --> DLQ\n    \n    CQ --> CEH\n    OQ --> OEH\n    \n    CEH --> NS\n    OEH --> NS\n    \n    NS --> EH\n    NS --> PH\n    NS --> WC\n```\n\nThis shows the complete event dependency graph with publishers, message routing, and consumer relationships.",
  "context_snippets": [
    {
      "content": "// • Emits events: `order-created`, `order-paid` (TODO)\n// • Communicates with notification-service",
      "file_path": "car-order-service/README.md",
      "line_range": [19, 20]
    },
    {
      "content": "// • Listens to events from car-listing-service and order-service (TODO)\n// • Can integrate with email/push notification providers",
      "file_path": "car-notification-service/README.md",
      "line_range": [19, 20]
    }
  ],
  "source_files": ["car-order-service/README.md", "car-notification-service/README.md"],
  "citations": [
    {
      "file_path": "car-order-service/README.md",
      "line": 19,
      "snippet_id": "event_publishing_deps"
    },
    {
      "file_path": "car-notification-service/README.md",
      "line": 19,
      "snippet_id": "event_consumption_deps"
    }
  ],
  "confidence_score": 0.91,
  "follow_up_prompts": [
    "How are event ordering dependencies managed?",
    "What happens with circular event dependencies?",
    "Show me the retry and error handling graph"
  ]
}
```

## 3. Technology Dependency Analysis

### Q5: Technology stack dependency matrix
**Question:** "What are the shared technology dependencies across all repositories and their version constraints?"

**Expected JSON Output:**
```json
{
  "answer": "Technology dependency matrix across repositories:\n\n**Shared Dependencies:**\n```\n| Technology | car-order-service | car-notification-service | car-web-client |\n|------------|------------------|-------------------------|----------------|\n| .NET 8     | ✓ Primary        | ✓ Primary               | ✗              |\n| TypeScript | ✗                | ✗                       | ✓ 86.1%        |\n| RabbitMQ   | ✓ Events         | ✓ Events                | ✗ (via WebSocket) |\n| PostgreSQL | ✓ Primary DB     | ✓ Optional              | ✗              |\n| MongoDB    | ✗                | ✓ Primary DB            | ✗              |\n| Swagger    | ✓ /swagger       | ✓ /swagger              | ✗              |\n| React      | ✗                | ✗                       | ✓ CRA          |\n```\n\n**Dependency Conflicts:**\n- Database: PostgreSQL vs MongoDB preferences\n- Message Queue: RabbitMQ vs Kafka options\n- Frontend: No direct backend dependency\n\n**Critical Path Dependencies:**\n1. RabbitMQ → All event-driven communication\n2. .NET 8 → Backend service compatibility\n3. HTTP REST → Frontend-backend integration",
  "context_snippets": [
    {
      "content": "## Tech Stack\n• .NET 8 Web API\n• PostgreSQL (or MySQL)\n• RabbitMQ (for event communication)",
      "file_path": "car-order-service/README.md",
      "line_range": [9, 11]
    },
    {
      "content": "TypeScript 86.1%\nHTML 9.0%\nCSS 4.9%",
      "file_path": "car-web-client/stats",
      "line_range": [1, 3]
    }
  ],
  "source_files": ["car-order-service/README.md", "car-web-client/stats"],
  "citations": [
    {
      "file_path": "car-order-service/README.md",
      "line": 9,
      "snippet_id": "tech_matrix"
    },
    {
      "file_path": "car-web-client/stats",
      "line": 1,
      "snippet_id": "frontend_dependencies"
    }
  ],
  "confidence_score": 0.92,
  "follow_up_prompts": [
    "How are version conflicts resolved?",
    "What are the upgrade paths for shared dependencies?",
    "Show me the deployment dependency graph"
  ]
}
```

### Q6: API dependency graph analysis
**Question:** "Show me the API dependencies and integration points between all repositories."

**Expected JSON Output:**
```json
{
  "answer": "API dependency graph and integration analysis:\n\n```mermaid\ngraph LR\n    subgraph \"External Dependencies\"\n        PG[Payment Gateway API]\n        EMAIL[Email Service API]\n        PUSH[Push Notification API]\n    end\n    \n    subgraph \"car-web-client APIs\"\n        WC[React Frontend]\n    end\n    \n    subgraph \"car-order-service APIs\"\n        OA1[GET /Order]\n        OA2[POST /Order]\n        OA3[PUT /Order/{id}/status]\n        OSW[Swagger /swagger]\n    end\n    \n    subgraph \"car-notification-service APIs\"\n        NA1[GET /Notification]\n        NA2[POST /Notification]\n        NA3[PUT /Notification/{id}/read]\n        NSW[Swagger /swagger]\n    end\n    \n    subgraph \"car-listing-service APIs\"\n        CA1[GET /Car]\n        CA2[POST /Car]\n        CA3[PUT /Car/{id}]\n        CSW[Swagger /swagger]\n    end\n    \n    WC -->|HTTP REST| OA1\n    WC -->|HTTP REST| OA2\n    WC -->|HTTP REST| NA1\n    WC -->|HTTP REST| NA2\n    WC -->|HTTP REST| CA1\n    \n    OA2 -.->|Verify Car| CA1\n    OA3 -->|Payment| PG\n    \n    NA2 -->|Send Email| EMAIL\n    NA2 -->|Send Push| PUSH\n    \n    OSW -.->|Documentation| WC\n    NSW -.->|Documentation| WC\n    CSW -.->|Documentation| WC\n```\n\nKey Integration Points:\n1. **Frontend → Backend**: REST API calls for all operations\n2. **Service → Service**: Cross-service verification calls\n3. **Backend → External**: Payment and notification providers\n4. **Documentation**: Swagger endpoints for API discovery",
  "context_snippets": [
    {
      "content": "// API Endpoints\n// • GET /Order - Get all orders\n// • GET /Order/{id} - Get order by id\n// • POST /Order - Create a new order\n// • PUT /Order/{id}/status - Update order status\n// • DELETE /Order/{id} - Delete an order",
      "file_path": "car-order-service/README.md",
      "line_range": [25, 30]
    }
  ],
  "source_files": ["car-order-service/README.md"],
  "citations": [
    {
      "file_path": "car-order-service/README.md",
      "line": 25,
      "snippet_id": "api_integration_points"
    }
  ],
  "confidence_score": 0.89,
  "follow_up_prompts": [
    "How are API versioning conflicts handled?",
    "What are the authentication dependencies?",
    "Show me the rate limiting and circuit breaker patterns"
  ]
}
```

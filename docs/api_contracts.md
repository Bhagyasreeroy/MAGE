# MAGE API Contracts

Base URL: `http://localhost:8000`

Auto-generated interactive docs: `http://localhost:8000/docs` (Swagger UI)

---

## Endpoints

### `GET /health`

Service liveness check.

**Response `200 OK`**

```json
{
  "status": "ok",
  "service": "MAGE Backend",
  "version": "0.1.0"
}
```

---

### `POST /analysis/run`

Trigger a full goal-conditioned EDA pipeline run.

**Request Body** (`application/json`)

```json
{
  "goal": "Identify the top factors driving customer churn.",
  "expertise_level": "intermediate",
  "dataset_name": "crm_export_2024.csv",
  "metadata": {
    "source": "upload",
    "rows": 50000
  }
}
```

| Field | Type | Required | Description |
|---|---|---|---|
| `goal` | `string` (5–2000 chars) | ✅ | Natural-language analytical goal |
| `expertise_level` | `"beginner"` \| `"intermediate"` \| `"expert"` | ❌ (default: `intermediate`) | Adapts recommendation verbosity |
| `dataset_name` | `string` \| `null` | ❌ | Name of the uploaded dataset |
| `metadata` | `object` | ❌ | Arbitrary key-value context |

**Response `200 OK`**

```json
{
  "goal": "Identify the top factors driving customer churn.",
  "expertise_level": "intermediate",
  "steps": [
    {
      "agent": "IngestionAgent",
      "status": "success",
      "output": {
        "schema": {},
        "row_count": 0,
        "quality_warnings": [],
        "message": "[STUB] IngestionAgent ran successfully."
      }
    },
    {
      "agent": "MiningAgent",
      "status": "success",
      "output": { "statistics": {}, "correlations": {}, "outliers": [], "patterns": [] }
    },
    {
      "agent": "VisualizationAgent",
      "status": "success",
      "output": { "viz_specs": [] }
    },
    {
      "agent": "RecommendationAgent",
      "status": "success",
      "output": { "recommendations": [], "rag_sources": [] }
    }
  ],
  "recommendations": [
    "[DUMMY] Recommendation 1 — placeholder until RAG is wired.",
    "[DUMMY] Recommendation 2 — placeholder until RAG is wired."
  ],
  "rag_sources": ["[DUMMY] knowledge_base/eda_best_practices.md"],
  "summary": "[DUMMY] Analysis of goal '...' completed in 4 steps."
}
```

**Error Responses**

| Code | Description |
|---|---|
| `422 Unprocessable Entity` | Request validation failed (e.g. goal too short) |
| `500 Internal Server Error` | Pipeline execution failed |

---

## Planned Endpoints (Future Milestones)

| Method | Path | Description |
|---|---|---|
| `POST` | `/upload` | Multipart file upload for datasets |
| `GET` | `/analysis/{run_id}` | Retrieve results for a past run |
| `GET` | `/analysis/{run_id}/viz` | Fetch VizSpec manifests for rendering |
| `GET` | `/kb/documents` | List knowledge-base documents |
| `POST` | `/kb/documents` | Add a document to the knowledge base |
| `DELETE` | `/kb/documents/{doc_id}` | Remove a document from the knowledge base |

---

## Schema Definitions

### `ExpertiseLevel`
```
"beginner" | "intermediate" | "expert"
```

### `StepResult`
```json
{
  "agent": "string",
  "status": "string",
  "output": {}
}
```

### `AnalysisResponse`
```json
{
  "goal": "string",
  "expertise_level": "ExpertiseLevel",
  "steps": ["StepResult"],
  "recommendations": ["string"],
  "rag_sources": ["string"],
  "summary": "string"
}
```

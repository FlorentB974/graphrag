# Potential Issues and Bad Practices

## Disabled TLS verification for OpenAI proxy
- Both embedding and LLM clients skip TLS certificate verification whenever `OPENAI_PROXY` is configured (`verify=False`). This invites man-in-the-middle attacks and makes it easy to leak prompts or credentials if traffic is intercepted. Prefer a trusted proxy certificate or leave verification enabled.

## Document preview trusts stored file paths
- `/api/documents/{document_id}/preview` streams files from whatever `file_path` is stored in Neo4j. The handler simply resolves relative paths against the project root and serves the file if it exists, without validating the path against an allowed directory or filename pattern. If the database is poisoned (e.g., via another bug or compromised admin credentials), an attacker could point `file_path` at arbitrary system files and have the API return them.

## Weak default Neo4j credentials in docker-compose
- The bundled `docker-compose.yml` boots Neo4j with static `neo4j/graphrag_password` credentials. Leaving these defaults in non-development environments exposes the graph database to trivial compromise. Require callers to set strong credentials (e.g., via env vars) or document that the password must be changed immediately.

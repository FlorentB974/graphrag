# GraphRAG Chat Experience (User-Facing)

A lightweight chat-only interface for the GraphRAG backend. This UI focuses on a single-page conversation experience without uploads or database management, making it suitable for end users while reusing the existing chat flow and streaming responses.

## Quickstart

1. Install dependencies:

```bash
npm install
```

2. Configure the backend endpoint (defaults to `http://localhost:8000`):

```bash
cp .env.example .env
# edit .env to set VITE_API_URL=http://your-api-host:8000
```

3. Run the development server on port 4000:

```bash
npm run dev
```

4. Build and preview:

```bash
npm run build
npm run preview
```

## Environment

- `VITE_API_URL` (optional): Base URL for the GraphRAG FastAPI backend. If omitted, `http://localhost:8000` is used.

## Features

- Clean chat-only surface with streaming answers
- Displays backend stages, quality scores, and cited sources
- Quick follow-up suggestions that keep the same conversation session
- "Start fresh" to reset the conversation while preserving the backend flow
- Health indicator that pings `/api/health`

## Project Structure

```
chat-frontend-lite/
├── src/
│   ├── App.tsx        # Chat experience
│   ├── App.css        # Component styling
│   ├── index.css      # Global styles
│   └── main.tsx       # React entrypoint
├── public/            # Static assets
├── vite.config.ts     # Vite configuration (ports set to 4000)
└── package.json
```

## Notes

- The UI streams from `POST /api/chat/query` and understands the existing SSE payloads (`token`, `stage`, `sources`, `follow_ups`, `quality_score`, `metadata`, `done`).
- No document upload or admin controls are exposed in this surface; it is designed purely for conversation.

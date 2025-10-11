# Implement Document View + Preview (Copilot Tasks)

This `TASKS.md` describes the work needed to implement a Document View in the frontend, plus required backend endpoints and tests. The goal: when a document is clicked in the Database tab, the app should replace the `ChatInterface` with a `DocumentView` that shows all document metadata and allows previewing the source document.

High level: frontend-only changes (components, store, API client), backend API endpoints (document metadata & preview), and tests + docs.

Summary of work items (detailed tasks follow):

- Add `DocumentView` and `DocumentPreview` components (frontend).
- Add store state to track `selectedDocumentId` and `activeView` (chat | document).
- Wire Database tab clicks to select a document and switch view.
- Add backend endpoints: `GET /documents/{id}` (metadata) and `GET /documents/{id}/preview` (preview URL or file streaming).
- Update frontend API client & types.
- Add tests (frontend store + backend endpoint) and update README/docs.

Project conventions and assumptions
- Frontend is in `frontend/` using React (Next.js app router under `src/`) + TypeScript + Tailwind.
- Global UI state is in `frontend/src/store/chatStore.ts` using Zustand.
- API helper is `frontend/src/lib/api.ts`.
- Backend is a Python FastAPI-style API inside `api/routers` or `api/main.py`.
- Files are stored locally under `data/` or behind a storage provider. The preview endpoint may redirect to a URL or stream a local file.

Design contract (API shapes)

GET /documents/{document_id}
- Response 200 JSON schema:
  {
    id: string,
    title?: string,
    file_name?: string,
    mime_type?: string,
    preview_url?: string,        # optional public URL or route to download/preview
    uploaded_at?: string,        # ISO timestamp
    uploader?: { id?: string, name?: string } | null,
    chunks: [
      {
        id: string | number,
        text: string,
        index?: number,
        offset?: number,
        score?: number | null
      }
    ],
    entities: [
      { type: string, text: string, count?: number, positions?: Array<number> }
    ],
    quality_scores?: any,        # object with quality metrics if available
    related_documents?: [
      { id: string, title?: string, link?: string }
    ],
    metadata?: object
  }

GET /documents/{document_id}/preview
- Response: either a redirect to a public URL or stream the file bytes with correct Content-Type.

Frontend changes (step-by-step)

1) Add store fields and actions
- File: `frontend/src/store/chatStore.ts`
- Add these state fields and actions:
  - activeView: 'chat' | 'document' (default: 'chat')
  - selectedDocumentId: string | null
  - selectDocument(id: string): sets selectedDocumentId and activeView = 'document'
  - clearSelectedDocument(): sets selectedDocumentId = null and activeView = 'chat'
  - setActiveView(view)

2) Add types
- Update `frontend/src/types/index.ts` (or create new types if needed) to include `Document`, `Chunk`, `Entity`, `RelatedDocument` interfaces matching API contract.

3) API client
- File: `frontend/src/lib/api.ts`
- Add functions:
  - getDocument(documentId: string): Promise<Document>
  - getDocumentPreview(documentId: string): Promise<{ preview_url: string } | Response>
- Use existing fetch wrapper used elsewhere (e.g., `api.get` / `api.post`) to keep auth/session.

4) DocumentView component
- New file: `frontend/src/components/Document/DocumentView.tsx`
- Responsibilities:
  - Read `selectedDocumentId` from store. If null, fallback to rendering `ChatInterface` or a message.
  - Fetch document metadata using `getDocument` on mount / when `selectedDocumentId` changes.
  - Render header with title, back button (calls `clearSelectedDocument()`), uploaded time, file name, mime type, uploader.
  - Show sections: Chunks (list with expand/collapse and copy-to-clipboard), Entities (grouped), Quality Scores (if present), Related Documents (clickable to navigate to their DocumentView), Metadata (JSON view).
  - Button to open `DocumentPreview` when `preview_url` is present.
  - Use existing Tailwind styles similar to `ChatInterface.tsx`.

5) DocumentPreview component
- New file: `frontend/src/components/Document/DocumentPreview.tsx`
- Props: { previewUrl: string, mimeType?: string, onClose?: () => void }
- Behavior:
  - For PDFs: use <iframe src={previewUrl} /> or <object>. For images: <img />. For other types: show a download link and an iframe fallback.
  - Provide toolbar with 'Open in new tab' and 'Close'.
  - Render as modal overlay or embedded section inside `DocumentView` depending on available space.

6) Wire DatabaseTab click behavior
- File: `frontend/src/components/Sidebar/DatabaseTab.tsx`
- Find the clickable document row or link that currently handles opening or selecting a document.
- Replace or augment the click handler to call store.selectDocument(docId). If the UI used navigation, keep route support as optional fallback.

7) Edge cases and UX notes
- If preview URL is absent but local file exists, call `GET /documents/{id}/preview` to attempt retrieving a temporary preview URL.
- Show loading skeleton while document metadata loads.
- Show friendly error on 404.
- Keep chat state intact when switching back to chat.

Backend changes (minimal, safe)

1) Add document metadata route
- Suggested file: `api/routers/database.py` or `api/routers/documents.py`.
- Route: `GET /documents/{document_id}`
- Implementation: query the internal document store (Neo4j or whatever is used in `core/graph_db.py`) for the document node and related chunk nodes, entities, relations, and metadata. Map to the contract and return JSON.

2) Add or reuse preview route
- Route: `GET /documents/{document_id}/preview`
- Implementation options:
  - If files are stored in `data/` or a public bucket, return redirect to the file URL.
  - Or stream the file using FastAPI's `FileResponse` with `media_type` inferred from `mime_type`.

3) Error handling
- 404 when document id not found
- 500 for unexpected errors

Tests

- Frontend: add Jest tests for `chatStore` new actions. Test selecting a document toggles activeView and sets `selectedDocumentId`.
- Backend: add tests to `api/tests/test_documents.py` checking that `GET /documents/{id}` returns the expected JSON shape for a seeded test document. Use existing test fixtures if present.

Acceptance criteria (how to verify)

- Click a document in `DatabaseTab` => `DocumentView` replaces `ChatInterface` while preserving session state.
- `DocumentView` shows header info, chunk list, entities, related docs, and a preview button when available.
- Clicking preview opens the document in an inline viewer or modal.
- Back button returns user to chat interface with previous messages intact.
- New endpoints return the documented JSON shapes. Tests exist for store and backend route.

Implementation checklist (ordered)

1. Update `frontend/src/store/chatStore.ts` with new state/actions.
2. Add TypeScript types in `frontend/src/types/index.ts`.
3. Add API client functions to `frontend/src/lib/api.ts`.
4. Create `frontend/src/components/Document/DocumentView.tsx` and `DocumentPreview.tsx`.
5. Wire `frontend/src/components/Sidebar/DatabaseTab.tsx` to `selectDocument` action.
6. Implement backend `GET /documents/{document_id}` and `/preview` routes under `api/routers`.
7. Add tests for store and backend endpoint.
8. Update `README.md` and `docs/` with a small usage note.

Developer notes and tips
- Keep front-end components small and reuse existing UI components where possible (buttons, badges, list items). Look at `ChatInterface.tsx` for style conventions.
- Use Tailwind utility classes; keep layout responsive and similar to the chat view.
- For previewing PDFs in Next.js, embedding via <iframe src={`/api/documents/${id}/preview`} /> works well and leverages backend preview endpoint for auth-controlled access.
- For large documents, lazy-load chunks or paginate if necessary; initial implementation can render all chunks and paginate later.

Files to create / modify (quick reference)

- frontend/src/store/chatStore.ts (modify)
- frontend/src/types/index.ts (modify)
- frontend/src/lib/api.ts (modify)
- frontend/src/components/Document/DocumentView.tsx (new)
- frontend/src/components/Document/DocumentPreview.tsx (new)
- frontend/src/components/Sidebar/DatabaseTab.tsx (modify)
- api/routers/documents.py (new) or api/routers/database.py (modify)
- api/tests/test_documents.py (new tests)
- TASKS.md (this file) added at repo root

If anything here is unclear or you want me to implement the first concrete changes (store + types + DocumentView skeleton), tell me which step to start with and I'll apply a patch.

---
Generated: Copilot TASKS for feature/new_ui branch

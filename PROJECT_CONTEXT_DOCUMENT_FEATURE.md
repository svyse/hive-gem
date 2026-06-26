# Project context document for code mode

Added a separate code-pipeline document option that lets a user start a code run from:

- a project path / sample_projects target
- a text or voice prompt
- an optional uploaded document used as one-off project context

This does **not** reuse or modify the existing `/api/uploads` document-ingestion path. The existing upload flow still chunks documents and stores them in Hive memory when `add_to_hive=true`.

## Implementation summary

- New backend endpoint: `POST /api/runs/with-document`
  - Multipart form fields: `project_path`, `prompt`, `copy_project_to_workspace`, `input_mode`, `file`
  - Extracts text with the same parser utilities, but does not call `ingest_upload`, does not chunk text, and does not create upload records.
- `RunManager.start_run(...)` now accepts optional `project_context_document`.
  - Saves the extracted text inside the target project at `.hive_project_context/uploaded_document_extracted.txt`.
  - Saves metadata at `.hive_project_context/uploaded_document_meta.json`.
  - Adds a bounded document excerpt and file path to the one run prompt so the code pipeline can use the document.
  - Stores only original prompt + document metadata in the existing run_prompt memory event, avoiding full document chunk storage through this path.
- Frontend code mode now has an “Optional project context document” file picker.
  - If selected, `createRunWithDocument` calls the new multipart endpoint.
  - If not selected, the existing `/api/runs` JSON flow is unchanged.

## Files changed

- `backend/app/api/routes.py`
- `backend/app/runtime/run_manager.py`
- `frontend/src/api/client.js`
- `frontend/src/components/PromptPanel.jsx`
- `frontend/src/App.jsx`

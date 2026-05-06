#!/usr/bin/env bash
set -euo pipefail

# Run backend and frontend in two terminals.
echo "Backend:  cd backend && uvicorn app.main:app --reload --port 8000"
echo "Frontend: cd frontend && npm install && npm run dev"

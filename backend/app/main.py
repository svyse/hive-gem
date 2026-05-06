from __future__ import annotations

import logging
from typing import Any

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router as api_router
from app.core.config import settings
from app.core.logging import configure_logging
from app.training.background import maybe_start_background_trainer
from app.workspace_modules.builtin import ensure_builtin_modules


configure_logging()
log = logging.getLogger(__name__)


def create_app() -> FastAPI:
    app = FastAPI(title="Agentic Hive Studio", version="0.1")

    # CORS for the Vite dev server (and any configured origins)
    allow = settings.cors_allow_origins or ["http://localhost:5173", "http://127.0.0.1:5173"]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allow,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"] ,
    )

    app.include_router(api_router, prefix="/api")

    @app.get("/")
    def root() -> dict[str, Any]:
        return {"ok": True, "service": "agentic-hive-backend"}

    @app.on_event("startup")
    async def _startup() -> None:
        # Ensure built-in workspace modules exist.
        # Best-effort: failures should never stop the API from starting.
        try:
            ensure_builtin_modules()
        except Exception as e:
            log.warning("workspace modules bootstrap failed: %s", e)

        # Starts the background training supervisor if enabled.
        # This is best-effort; failures should never stop the API from starting.
        maybe_start_background_trainer()

    return app


app = create_app()


if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="127.0.0.1",
        port=int(settings.backend_port),
        reload=True,
        log_level=(settings.log_level or "info").lower(),
    )

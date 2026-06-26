import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, RedirectResponse

from shared.config import API_VERSION, APP_NAME
from shared.db import check_db, init_db
from app.routes.score import router as score_router
from app.routes.ui import router as ui_router

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    if os.getenv("TESTING") != "1":
        logger.info("Initializing database...")
        init_db()
    yield


app = FastAPI(
    title=APP_NAME,
    version=API_VERSION,
    description="MVP сервиса кредитного скоринга (Home Credit)",
    lifespan=lifespan,
)

app.include_router(ui_router)
app.include_router(score_router)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    if request.url.path == "/score" and request.method == "POST":
        accept = request.headers.get("accept", "")
        if "text/html" in accept or "multipart" in request.headers.get("content-type", ""):
            return RedirectResponse(url="/?error=form", status_code=303)
    return JSONResponse(status_code=422, content={"detail": exc.errors()})


@app.get("/health")
async def health():
    db_ok = check_db()
    return JSONResponse(
        {
            "status": "ok" if db_ok else "degraded",
            "database": "connected" if db_ok else "disconnected",
        }
    )

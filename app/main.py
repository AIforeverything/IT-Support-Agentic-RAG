from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.api.routes import router
from app.core.config import get_settings, BASE_PATH
from app.core.logging import configure_logging
from app.services.audit import init_db

configure_logging()
settings = get_settings()
init_db()

app = FastAPI(title=settings.APP_NAME, version="1.0.0")

# adding router
app.include_router(router=router)
app.mount("/static", StaticFiles(directory=str(BASE_PATH / "static")), name="static")
templates = Jinja2Templates(directory=str(BASE_PATH / "templates"))


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(
        "index.html", {"request": request, "app_name": settings.APP_NAME}
    )

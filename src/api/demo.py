"""Browser demo page served at ``/``."""

from pathlib import Path

from fastapi.responses import HTMLResponse

_STATIC_DIR = Path(__file__).resolve().parent / "static"


def demo_page() -> HTMLResponse:
    return HTMLResponse(content=(_STATIC_DIR / "index.html").read_text(encoding="utf-8"))

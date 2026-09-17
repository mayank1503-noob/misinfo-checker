from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from backend.analyzers.packet import from_text, from_link


app = FastAPI(
    title="Misinformation Detection API",
    version="1.0.0"
)


class TextRequest(BaseModel):
    text: str


class LinkRequest(BaseModel):
    url: str


@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "misinformation-detection-api"
    }


@app.post("/check/text")
def check_text(request: TextRequest):
    text = request.text.strip()

    if not text:
        raise HTTPException(
            status_code=422,
            detail="Text cannot be empty"
        )

    if len(text) > 10000:
        raise HTTPException(
            status_code=413,
            detail="Text is too long. Maximum length is 10,000 characters."
        )

    try:
        packet = from_text(text)

        return {
            "packet": packet,
            "results": []
        }

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Text analysis failed: {str(error)}"
        )


@app.post("/check/link")
def check_link(request: LinkRequest):
    url = request.url.strip()

    if not url:
        raise HTTPException(
            status_code=422,
            detail="URL cannot be empty"
        )

    if not (
        url.startswith("http://")
        or url.startswith("https://")
    ):
        raise HTTPException(
            status_code=422,
            detail="URL must start with http:// or https://"
        )

    try:
        packet = from_link(url)

        return {
            "packet": packet,
            "results": []
        }

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Link analysis failed: {str(error)}"
        )
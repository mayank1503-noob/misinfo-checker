import trafilatura
from htmldate import find_date


def extract(url):
    html = trafilatura.fetch_url(url)

    if not html:
        return "", None

    text = trafilatura.extract(html) or ""

    try:
        date = find_date(html)
    except Exception:
        date = None

    return text, date
def is_supported_message(text):
    if not text:
        return False

    text = text.strip()

    return len(text) > 0


def contains_url(text):
    if not text:
        return False

    return (
        "http://" in text
        or "https://" in text
        or "www." in text
    )


def classify_message(text):
    if not is_supported_message(text):
        return "empty"

    if contains_url(text):
        return "link"

    return "text"
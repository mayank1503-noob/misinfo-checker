from . import link
from . import image
from . import video


def _packet(input_type, text="", source_date=None, images=None):
    return {
        "input_type": input_type,
        "text": text or "",
        "source_date": source_date,
        "images": images or [],
    }


def from_text(text):
    return _packet("text", text)


def from_link(url):
    text, date = link.extract(url)
    return _packet("link", text, date)


def from_image(path, caption=""):
    return _packet(
        "image",
        caption,
        images=[image.analyze(path)]
    )


def from_video(path, caption=""):
    transcript, frames = video.analyze(path)

    text = (caption + "\n" + transcript).strip()

    return _packet(
        "video",
        text,
        images=frames
    )


def from_video_url(url):
    return from_video(video.download(url))
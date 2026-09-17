from . import link
from . import image
from . import video


# Where a packet lists the temp paths it owns. Leading underscore: it is
# bookkeeping, and `pipeline._public_packet` strips it from responses.
TEMP_PATHS = "_temp_paths"


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


def from_video(path, caption="", workdir=None):
    """
    A packet from a video file.

    The keyframes live in a temp directory that stages 4-6 still read by
    path, so the packet carries it under TEMP_PATHS and whoever built the
    packet calls `cleanup(packet)` once the pipeline is done with it.
    """
    transcript, frames, workdir = video.analyze(path, workdir=workdir)

    text = (caption + "\n" + transcript).strip()

    packet = _packet(
        "video",
        text,
        images=frames
    )

    packet[TEMP_PATHS] = [workdir]

    return packet


def from_video_url(url, caption=""):
    """The same, for a video we fetch ourselves; the download is temp too."""
    workdir = video.workspace()

    try:
        return from_video(
            video.download(url, workdir=workdir),
            caption,
            workdir=workdir,
        )
    except BaseException:
        video.discard(workdir)
        raise


def cleanup(packet):
    """
    Delete the temp files a packet owns.

    Safe on any packet and safe to call twice: text, link and image
    packets own nothing.
    """
    for path in (packet or {}).pop(TEMP_PATHS, None) or []:
        video.discard(path)
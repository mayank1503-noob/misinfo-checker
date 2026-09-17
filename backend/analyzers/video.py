import logging
import os
import shutil
import subprocess
import tempfile

from . import models
from . import image


log = logging.getLogger(__name__)


def workspace():
    """
    A private directory for everything one video spills onto disk.

    Whoever calls this owns the directory and must hand it to `discard`
    when the analysis that reads from it is finished; the keyframes
    outlive `analyze` because stage 5 (CLIP, reverse search) opens them
    again by path.
    """
    return tempfile.mkdtemp(prefix="misinfo-video-")


def discard(path):
    """Remove a temp file or directory. Cleanup never raises."""
    if not path:
        return

    try:
        if os.path.isdir(path):
            shutil.rmtree(path, ignore_errors=True)
        elif os.path.exists(path):
            os.remove(path)
    except OSError as error:
        log.warning("could not remove temp path %s: %s", path, error)


def extract_audio(video_path, workdir=None):
    """
    The video's audio as 16 kHz mono wav, in a file only we can read.

    `mkstemp` rather than `mktemp`: the latter only reserves a name, so
    anything else on the box can create that path first and have ffmpeg
    write through its symlink. The caller removes the file.
    """
    handle, audio_path = tempfile.mkstemp(suffix=".wav", dir=workdir)
    os.close(handle)

    command = [
        "ffmpeg",
        "-y",
        "-i",
        video_path,
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        audio_path,
    ]

    try:
        subprocess.run(
            command,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except BaseException:
        # A failed ffmpeg still leaves the empty (or half-written) file.
        discard(audio_path)
        raise

    return audio_path


def transcribe(video_path, workdir=None):
    audio_path = extract_audio(video_path, workdir=workdir)

    try:
        model = models.whisper_model()
        result = model.transcribe(audio_path)

        return result.get("text", "").strip()

    finally:
        discard(audio_path)


def extract_keyframes(video_path, interval=5, max_frames=6, workdir=None):
    """
    One frame every `interval` seconds, analyzed.

    Returns `(frames, frame_dir)`. The frames keep their paths, so the
    directory has to stay until the pipeline is done with them.
    """
    frame_dir = tempfile.mkdtemp(prefix="frames-", dir=workdir)

    pattern = os.path.join(frame_dir, "frame_%03d.jpg")

    command = [
        "ffmpeg",
        "-y",
        "-i",
        video_path,
        "-vf",
        f"fps=1/{interval}",
        "-frames:v",
        str(max_frames),
        pattern,
    ]

    try:
        subprocess.run(
            command,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        frames = []

        for filename in sorted(os.listdir(frame_dir)):
            path = os.path.join(frame_dir, filename)

            try:
                frames.append(
                    image.analyze(path)
                )
            except Exception:
                pass

    except BaseException:
        discard(frame_dir)
        raise

    return frames, frame_dir


def analyze(video_path, workdir=None):
    """
    Transcript plus keyframes.

    Returns `(transcript, frames, workdir)`: the third value is the
    directory holding the frame images, which the caller discards once
    nothing reads them any more.
    """
    own_workdir = workdir is None
    workdir = workdir or workspace()

    try:
        transcript = transcribe(video_path, workdir=workdir)
        frames, _ = extract_keyframes(video_path, workdir=workdir)
    except BaseException:
        if own_workdir:
            discard(workdir)
        raise

    return transcript, frames, workdir


def download(url, workdir=None):
    """Fetch a video into `workdir` (a fresh workspace by default)."""
    own_workdir = workdir is None
    workdir = workdir or workspace()

    output_path = os.path.join(
        workdir,
        "video.%(ext)s"
    )

    import yt_dlp

    options = {
        "outtmpl": output_path,
        "format": "best[filesize<50M]/best",
        "noplaylist": True,
    }

    try:
        with yt_dlp.YoutubeDL(options) as downloader:
            info = downloader.extract_info(
                url,
                download=True
            )

            return downloader.prepare_filename(info)
    except BaseException:
        if own_workdir:
            discard(workdir)
        raise

import os
import subprocess
import tempfile

from . import models
from . import image


def extract_audio(video_path):
    audio_path = tempfile.mktemp(suffix=".wav")

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

    subprocess.run(
        command,
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    return audio_path


def transcribe(video_path):
    audio_path = extract_audio(video_path)

    try:
        model = models.whisper_model()
        result = model.transcribe(audio_path)

        return result.get("text", "").strip()

    finally:
        if os.path.exists(audio_path):
            os.remove(audio_path)


def extract_keyframes(video_path, interval=5, max_frames=6):
    output_dir = tempfile.mkdtemp()

    pattern = os.path.join(output_dir, "frame_%03d.jpg")

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

    subprocess.run(
        command,
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    frames = []

    for filename in sorted(os.listdir(output_dir)):
        path = os.path.join(output_dir, filename)

        try:
            frames.append(
                image.analyze(path)
            )
        except Exception:
            pass

    return frames


def analyze(video_path):
    transcript = transcribe(video_path)
    frames = extract_keyframes(video_path)

    return transcript, frames


def download(url):
    output_dir = tempfile.mkdtemp()

    output_path = os.path.join(
        output_dir,
        "video.%(ext)s"
    )

    import yt_dlp

    options = {
        "outtmpl": output_path,
        "format": "best[filesize<50M]/best",
        "noplaylist": True,
    }

    with yt_dlp.YoutubeDL(options) as downloader:
        info = downloader.extract_info(
            url,
            download=True
        )

        return downloader.prepare_filename(info)
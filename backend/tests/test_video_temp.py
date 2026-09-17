"""
The temp files a video analysis leaves behind.

ffmpeg, Whisper and the image models are all stubbed: what is under test
is who creates temp paths and who removes them, not what the models say
about the pixels. The rule the tests pin down is that a video check
leaves nothing on disk afterwards, keeps the keyframes alive while the
pipeline still reads them by path, and never touches the input video.
"""

import os
import subprocess

import pytest

from backend.analyzers import packet as packet_module
from backend.analyzers import video


def fake_ffmpeg(frames=2):
    """
    Stand in for `subprocess.run`, writing what ffmpeg would write.

    An `-i <input> ... <output>` command ends in the output path (or an
    output pattern, for keyframes), which is all the stub needs.
    """
    def run(command, **kwargs):
        output = command[-1]

        if "%03d" in output:
            for index in range(1, frames + 1):
                with open(output.replace("%03d", f"{index:03d}"), "wb") as handle:
                    handle.write(b"jpeg")
        else:
            with open(output, "wb") as handle:
                handle.write(b"wav")

        return subprocess.CompletedProcess(command, 0)

    return run


@pytest.fixture
def stub_video(monkeypatch):
    """ffmpeg, Whisper and the per-frame image analysis, all faked."""
    monkeypatch.setattr(subprocess, "run", fake_ffmpeg())

    class Whisper:
        def transcribe(self, path):
            assert os.path.exists(path), "whisper got a deleted audio file"

            return {"text": "  the bridge collapsed yesterday  "}

    monkeypatch.setattr(video.models, "whisper_model", lambda: Whisper())
    monkeypatch.setattr(
        video.image,
        "analyze",
        lambda path, frame_time=None: {"path": path, "ocr_text": "", "embedding": []},
    )


@pytest.fixture
def source_video(tmp_path):
    path = tmp_path / "forward.mp4"
    path.write_bytes(b"not really a video")

    return str(path)


def test_extract_audio_is_not_a_predictable_name(stub_video, source_video, tmp_path):
    """
    The wav is a file we created, not a name we reserved.

    `tempfile.mktemp` handed ffmpeg a path anything else on the box could
    win the race for; `mkstemp` hands back a file that already exists and
    is ours.
    """
    workdir = str(tmp_path / "work")
    os.makedirs(workdir)

    audio_path = video.extract_audio(source_video, workdir=workdir)

    assert os.path.exists(audio_path)
    assert os.path.dirname(audio_path) == workdir

    video.discard(audio_path)


def test_transcribe_removes_the_audio(stub_video, source_video):
    before = video.workspace()

    try:
        text = video.transcribe(source_video, workdir=before)

        assert text == "the bridge collapsed yesterday"
        assert os.listdir(before) == [], "the wav outlived the transcription"
    finally:
        video.discard(before)


def test_failed_ffmpeg_leaves_no_wav(monkeypatch, source_video, tmp_path):
    def boom(command, **kwargs):
        # ffmpeg writes the output file, then dies on the input.
        with open(command[-1], "wb") as handle:
            handle.write(b"")

        raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(subprocess, "run", boom)

    workdir = str(tmp_path / "work")
    os.makedirs(workdir)

    with pytest.raises(subprocess.CalledProcessError):
        video.extract_audio(source_video, workdir=workdir)

    assert os.listdir(workdir) == []


def test_analyze_keeps_the_frames_until_the_caller_is_done(stub_video, source_video):
    transcript, frames, workdir = video.analyze(source_video)

    try:
        assert transcript == "the bridge collapsed yesterday"
        assert len(frames) == 2

        # Stage 5 opens these again by path; they have to still be there.
        for frame in frames:
            assert os.path.exists(frame["path"])
    finally:
        video.discard(workdir)

    assert not os.path.exists(workdir)


def test_packet_owns_its_temp_directory(stub_video, source_video):
    packet = packet_module.from_video(source_video, caption="seen in a forward")

    workdirs = packet[packet_module.TEMP_PATHS]

    assert packet["text"].startswith("seen in a forward")
    assert all(os.path.isdir(path) for path in workdirs)

    packet_module.cleanup(packet)

    assert not any(os.path.exists(path) for path in workdirs)
    assert packet_module.TEMP_PATHS not in packet

    # Cleanup twice is a no-op, not an error.
    packet_module.cleanup(packet)


def test_the_input_video_survives(stub_video, source_video):
    packet = packet_module.from_video(source_video)

    packet_module.cleanup(packet)

    assert os.path.exists(source_video), "we deleted the caller's video"


def test_analyze_video_cleans_up_even_when_the_pipeline_fails(
    monkeypatch, stub_video, source_video
):
    from backend import pipeline

    seen = {}

    def explode(packet, **kwargs):
        seen["workdirs"] = list(packet[packet_module.TEMP_PATHS])

        raise RuntimeError("stage 3 fell over")

    monkeypatch.setattr(pipeline, "analyze", explode)

    with pytest.raises(RuntimeError):
        pipeline.analyze_video(source_video)

    assert seen["workdirs"]
    assert not any(os.path.exists(path) for path in seen["workdirs"])


def test_the_response_packet_has_no_temp_bookkeeping():
    from backend.pipeline import _public_packet

    public = _public_packet(
        {
            "input_type": "video",
            "text": "hi",
            "images": [{"path": "a.jpg", "embedding": [0.1]}],
            packet_module.TEMP_PATHS: ["/tmp/misinfo-video-abc"],
        }
    )

    assert packet_module.TEMP_PATHS not in public
    assert "embedding" not in public["images"][0]

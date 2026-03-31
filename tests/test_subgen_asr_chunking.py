import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from subgen.cli import (
    DEEPGRAM_SEGMENT_THRESHOLD_SECONDS,
    SubtitleEntry,
    _deepgram_listen_bytes,
    _merge_deepgram_chunk_entries,
    _plan_deepgram_segments,
    _transcribe_deepgram_window,
    transcribe_with_deepgram,
)


def test_plan_deepgram_segments_avoids_tiny_tail():
    segments = _plan_deepgram_segments(
        7200.0,
        chunk_duration=1200.0,
        overlap=2.0,
    )

    assert segments[0] == (0.0, 1200.0)
    assert segments[-1][1] == 7200.0
    assert all(end > start for start, end in segments)
    assert all((end - start) >= 60.0 for start, end in segments)
    assert DEEPGRAM_SEGMENT_THRESHOLD_SECONDS < 20 * 60


def test_merge_deepgram_chunk_entries_deduplicates_overlap():
    existing = [
        SubtitleEntry(index=1, start=118.0, end=120.0, text="hello world"),
    ]
    incoming = [
        SubtitleEntry(index=1, start=119.2, end=120.4, text="hello world"),
        SubtitleEntry(index=2, start=120.5, end=122.0, text="next line"),
    ]

    merged = _merge_deepgram_chunk_entries(
        existing,
        incoming,
        overlap_start=118.0,
        overlap_seconds=2.0,
    )

    assert [entry.text for entry in merged] == ["hello world", "next line"]
    assert [entry.index for entry in merged] == [1, 2]


def test_transcribe_with_deepgram_segments_long_media(monkeypatch, tmp_path):
    video_path = tmp_path / "long.mp4"
    video_path.write_bytes(b"fake-video")

    monkeypatch.setattr("subgen.cli._probe_media_duration", lambda _path: 3600.0)
    monkeypatch.setattr(
        "subgen.cli._plan_deepgram_segments",
        lambda _duration: [(0.0, 1200.0), (1198.0, 2400.0), (2398.0, 3600.0)],
    )
    monkeypatch.setattr(
        "subgen.cli._extract_audio_segment_bytes",
        lambda **_kwargs: b"chunk-audio",
    )

    payloads = [{"chunk": 1}, {"chunk": 2}, {"chunk": 3}]

    def fake_listen_bytes(**_kwargs):
        return payloads.pop(0)

    monkeypatch.setattr("subgen.cli._deepgram_listen_bytes", fake_listen_bytes)

    def fake_payload_to_entries(*, payload, max_segment_duration, max_segment_chars, time_offset):
        assert max_segment_duration == 2.2
        assert max_segment_chars == 28
        if payload["chunk"] == 1:
            return [
                SubtitleEntry(index=1, start=0.0, end=3.0, text="first"),
                SubtitleEntry(index=2, start=1198.2, end=1200.0, text="bridge"),
            ], "en", 0.7
        if payload["chunk"] == 2:
            return [
                SubtitleEntry(index=1, start=1198.3, end=1200.2, text="bridge"),
                SubtitleEntry(index=2, start=1200.2, end=1204.0, text="middle"),
            ], "en", 0.8
        return [
            SubtitleEntry(index=1, start=2398.1, end=2401.0, text="tail"),
        ], "en", 0.9

    monkeypatch.setattr("subgen.cli._deepgram_payload_to_entries", fake_payload_to_entries)

    progress_updates = []

    def on_progress(task, current, total, label):
        progress_updates.append((task, current, total, label))

    entries = transcribe_with_deepgram(
        video_path=video_path,
        source_language=None,
        max_segment_duration=2.2,
        max_segment_chars=28,
        api_key="test-key",
        model_name="nova-3",
        progress_label="demo.mp4",
        progress_callback=on_progress,
    )

    assert [entry.text for entry in entries] == ["first", "bridge", "middle", "tail"]
    assert entries[-1].index == 4
    assert progress_updates[0] == ("asr", 0.0, 3600.0, "demo.mp4")
    assert progress_updates[-1] == ("asr", 3600.0, 3600.0, "demo.mp4")


def test_transcribe_deepgram_window_splits_again_on_timeout(monkeypatch, tmp_path):
    video_path = tmp_path / "timeout.mp4"
    video_path.write_bytes(b"fake-video")

    monkeypatch.setattr(
        "subgen.cli._extract_audio_segment_bytes",
        lambda **_kwargs: b"chunk-audio",
    )

    attempted = []

    def fake_listen_bytes(**kwargs):
        attempted.append(kwargs)
        if len(attempted) == 1:
            raise RuntimeError("Deepgram 请求失败: timeout")
        return {"ok": len(attempted)}

    monkeypatch.setattr("subgen.cli._deepgram_listen_bytes", fake_listen_bytes)

    def fake_payload_to_entries(*, payload, max_segment_duration, max_segment_chars, time_offset):
        assert max_segment_duration == 2.2
        assert max_segment_chars == 28
        return [
            SubtitleEntry(index=1, start=time_offset, end=time_offset + 1.0, text=f"chunk-{payload['ok']}"),
        ], "en", 0.9

    monkeypatch.setattr("subgen.cli._deepgram_payload_to_entries", fake_payload_to_entries)

    entries, detected_lang, confidence = _transcribe_deepgram_window(
        video_path=video_path,
        start=0.0,
        end=240.0,
        source_language=None,
        api_key="test-key",
        model_name="nova-3",
        max_segment_duration=2.2,
        max_segment_chars=28,
    )

    assert len(attempted) == 3
    assert [entry.text for entry in entries] == ["chunk-2", "chunk-3"]
    assert detected_lang == "en"
    assert confidence == 0.9


def test_deepgram_listen_bytes_retries_ssl_eof(monkeypatch):
    monkeypatch.setattr("subgen.cli.time.sleep", lambda _seconds: None)

    attempts = {"count": 0}

    class _Resp:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return b'{"results":{"channels":[{"alternatives":[{"transcript":"ok"}]}]}}'

    def fake_urlopen(_req, timeout):
        assert timeout == 600
        attempts["count"] += 1
        if attempts["count"] < 3:
            raise RuntimeError(
                "<urlopen error [SSL: UNEXPECTED_EOF_WHILE_READING] EOF occurred in violation of protocol (_ssl.c:1016)>"
            )
        return _Resp()

    monkeypatch.setattr("subgen.cli.urlrequest.urlopen", fake_urlopen)

    payload = _deepgram_listen_bytes(
        media_bytes=b"audio",
        content_type="audio/flac",
        source_language=None,
        api_key="test-key",
        model_name="nova-3",
    )

    assert attempts["count"] == 3
    assert payload["results"]["channels"][0]["alternatives"][0]["transcript"] == "ok"


def test_transcribe_deepgram_window_splits_again_on_ssl_eof(monkeypatch, tmp_path):
    video_path = tmp_path / "ssl.mp4"
    video_path.write_bytes(b"fake-video")

    monkeypatch.setattr("subgen.cli._extract_audio_segment_bytes", lambda **_kwargs: b"chunk-audio")

    attempted = []

    def fake_listen_bytes(**kwargs):
        attempted.append(kwargs)
        if len(attempted) == 1:
            raise RuntimeError(
                "Deepgram 请求失败: <urlopen error [SSL: UNEXPECTED_EOF_WHILE_READING] EOF occurred in violation of protocol (_ssl.c:1016)>"
            )
        return {"ok": len(attempted)}

    monkeypatch.setattr("subgen.cli._deepgram_listen_bytes", fake_listen_bytes)

    def fake_payload_to_entries(*, payload, max_segment_duration, max_segment_chars, time_offset):
        return [
            SubtitleEntry(index=1, start=time_offset, end=time_offset + 1.0, text=f"ssl-{payload['ok']}"),
        ], "en", 0.95

    monkeypatch.setattr("subgen.cli._deepgram_payload_to_entries", fake_payload_to_entries)

    entries, detected_lang, confidence = _transcribe_deepgram_window(
        video_path=video_path,
        start=0.0,
        end=240.0,
        source_language=None,
        api_key="test-key",
        model_name="nova-3",
        max_segment_duration=2.2,
        max_segment_chars=28,
    )

    assert len(attempted) == 3
    assert [entry.text for entry in entries] == ["ssl-2", "ssl-3"]
    assert detected_lang == "en"
    assert confidence == 0.95

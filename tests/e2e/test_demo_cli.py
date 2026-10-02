"""`python scripts/run_demo.py` is the one command that proves the slice works."""

from __future__ import annotations

import json
import subprocess
import sys
import wave

from tests.conftest import REPO_ROOT


def test_demo_script_runs_to_completion(tmp_path) -> None:
    proc = subprocess.run([sys.executable, "scripts/run_demo.py", "--data-dir", str(tmp_path)], cwd=REPO_ROOT,
                          capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    out = proc.stdout
    for expected in ("task_id:", "final status:  COMPLETED", "workflow steps:", "generated artifacts:",
                     "learner mastery changes:", "token usage:", "estimated cost:", "actual cost:",
                     "WAITING for diagnostic round 1", "WAITING for diagnostic round 2",
                     "review:        APPROVED after 1 revision(s)"):
        assert expected in out, expected
    assert (tmp_path / "teaching_agent.db").exists()
    assert any((tmp_path / "objects").rglob("v1.json"))


def test_evaluation_demo_script_runs_to_completion(tmp_path) -> None:
    proc = subprocess.run([sys.executable, "scripts/run_evaluation_demo.py", "--data-dir", str(tmp_path)],
                          cwd=REPO_ROOT, capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    out = proc.stdout
    for expected in ("final status:  COMPLETED", "lesson completed:", "assessment generated:",
                     "task is WAITING for assessment_answers", "answers submitted; task status: COMPLETED",
                     "evaluation result:", "mastery before -> after:", "remaining gaps:  es.football.opinions",
                     "next recommendation: reteach"):
        assert expected in out, expected
    assert any((tmp_path / "objects").rglob("learner_evaluation/v1.json"))


def test_research_demo_script_runs_to_completion(tmp_path) -> None:
    proc = subprocess.run([sys.executable, "scripts/run_research_demo.py", "--data-dir", str(tmp_path)],
                          cwd=REPO_ROOT, capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    out = proc.stdout
    for expected in ("1. Research request", "2. Generated queries", "3. Mock search results", "4. Selected sources",
                     "5. Extracted evidence", "6. Citations", "7. ResearchBundle", "8. Lesson generated from the bundle",
                     "status:       complete", "reliability 0.30 is below 0.50", "duplicate: https://www.sports-news",
                     "[c6] -> ev6 ->", "search:mock"):
        assert expected in out, expected
    assert any((tmp_path / "objects").rglob("research_bundle/v1.json"))


def test_presentation_demo_script_writes_a_real_pptx(tmp_path) -> None:
    out_file = tmp_path / "lesson.pptx"
    proc = subprocess.run([sys.executable, "scripts/run_presentation_demo.py", "--data-dir", str(tmp_path / "data"),
                           "--out", str(out_file)], cwd=REPO_ROOT, capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    out = proc.stdout
    for expected in ("1. Completed lesson", "2. ResearchBundle", "3. IMAGE_ASSET artifacts", "4. SlideDeckPlan",
                     "5. Validation", "valid=True", "6. Presentation build", "7. Render PPTX",
                     "presentation.artifact_created", "8. Stored PRESENTATION artifact",
                     "application/vnd.openxmlformats-officedocument.presentationml.presentation",
                     "bytes match: True", "9. The .pptx opened with python-pptx", "pictures: image_v1_photo"):
        assert expected in out, expected
    from pptx import Presentation

    assert len(Presentation(str(out_file)).slides) >= 5
    assert any((tmp_path / "data" / "objects").rglob("*.pptx"))


def test_audio_demo_script_writes_real_wav_files_and_a_timeline(tmp_path) -> None:
    out_dir = tmp_path / "narration"
    proc = subprocess.run([sys.executable, "scripts/run_audio_demo.py", "--data-dir", str(tmp_path / "data"),
                           "--out-dir", str(out_dir)], cwd=REPO_ROOT, capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    out = proc.stdout
    for expected in ("1. Approved lesson", "2. Presentation", "3. AudioPlan", "4. Validation", "valid=True",
                     "5. Mock TTS", "6. Audio validation", "rejected: 0", "7. AUDIO_ASSET artifacts",
                     "bytes match: True", "8. PresentationTimeline", "9. Slide timings", "Slide  1 s01",
                     "10. Audio duration", "11. Artifact references", "PRESENTATION_TIMELINE",
                     "narration status: complete"):
        assert expected in out, expected
    wavs = sorted(out_dir.glob("*.wav"))
    assert wavs
    for path in wavs:
        with wave.open(str(path)) as wav:
            assert wav.getnframes() > 0 and wav.getframerate() == 16000
    timeline = json.loads((out_dir / "presentation_timeline.json").read_text(encoding="utf-8"))
    assert timeline["slides"][0]["start_time"] == 0.0 and timeline["duration"] > 0
    assert len(timeline["segments"]) == len(wavs)


def test_video_demo_script_writes_a_real_playable_mp4(tmp_path) -> None:
    """Lesson -> Presentation -> Visual assets -> Audio assets -> PresentationTimeline -> VideoPlan -> MP4 -> VIDEO
    artifact, with the real FFmpeg composer (at 640x360 to keep the suite quick; the demo defaults to 1080p)."""
    out_dir = tmp_path / "video"
    proc = subprocess.run([sys.executable, "scripts/run_video_demo.py", "--data-dir", str(tmp_path / "data"),
                           "--out-dir", str(out_dir), "--width", "640", "--height", "360"],
                          cwd=REPO_ROOT, capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    out = proc.stdout
    for expected in ("1. Approved lesson", "2. PRESENTATION artifact", "3. IMAGE_ASSET artifacts",
                     "4. AUDIO_ASSET artifacts", "5. PresentationTimeline", "6. VideoPlan", "7. VideoPlan validation",
                     "8. Composition", "9. MP4 validation", "10. VIDEO artifact", "11. Result",
                     "valid=True by video-plan-validator/1", "valid=True by video-validator/1+ffprobe/1",
                     "composer=ffmpeg-composer/1", "resolution:    640x360", "fps:           30",
                     "audio stream:  aac 48000 Hz, 2 channels", "cues burned in (yes)", "(matches artifact: True)",
                     "audio: silence"):
        assert expected in out, expected
    data = json.loads(subprocess.run(["ffprobe", "-v", "error", "-print_format", "json", "-show_format",
                                      "-show_streams", str(out_dir / "lesson.mp4")],
                                     capture_output=True, text=True, check=True).stdout)
    timeline = json.loads(next((tmp_path / "data" / "objects").rglob("presentation_timeline/v1.json")).read_text())
    assert abs(float(data["format"]["duration"]) - timeline["duration"]) <= 0.1
    assert {s["codec_type"] for s in data["streams"]} == {"video", "audio"}
    assert (out_dir / "lesson.vtt").read_text(encoding="utf-8").startswith("WEBVTT")
    assert any((tmp_path / "data" / "objects").rglob("*.mp4"))

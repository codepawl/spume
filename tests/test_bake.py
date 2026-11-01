# SPDX-License-Identifier: Apache-2.0
"""One small bake on the CPU, run through the real command line, so the tests need no GPU."""

import json
import subprocess
import sys

import pytest

FRAMES = 4


@pytest.fixture(scope="module")
def bake_folder(tmp_path_factory):
    folder = tmp_path_factory.mktemp("bake")
    command = [
        sys.executable, "-m", "spume",
        "--device", "cpu",
        "--out", str(folder),
        "--cell", "0.2",
        "--width", "3",
        "--height", "3",
        "--frames", str(FRAMES),
        "--mesh-cell", "0.2",
        "--speed", "3",
        "--opening", "0.5",
        "--liquid-box", "0.5", "0", "0.5", "2.5", "1", "2.5",
    ]
    subprocess.run(command, check=True, capture_output=True, text=True)
    return folder


def test_writes_one_binary_gltf_per_frame(bake_folder):
    frames = sorted(bake_folder.glob("frame_*.glb"))
    assert len(frames) == FRAMES
    for frame in frames:
        assert frame.read_bytes()[:4] == b"glTF"


def test_a_liquid_box_gives_a_surface_with_triangles(bake_folder):
    largest_frame = max(frame.stat().st_size for frame in bake_folder.glob("frame_*.glb"))
    assert largest_frame > 1000


def test_metadata_and_report_describe_the_bake(bake_folder):
    meta = json.loads((bake_folder / "meta.json").read_text())
    report = json.loads((bake_folder / "report.json").read_text())
    assert len(meta["headHeights"]) == FRAMES
    assert report["liquid_particles"] > 0
    assert report["total_seconds"] > 0


def test_an_unknown_device_stops_with_a_clear_message(tmp_path):
    command = [sys.executable, "-m", "spume", "--device", "cuda:99", "--out", str(tmp_path)]
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode != 0
    assert "cuda:99 was not found" in result.stderr

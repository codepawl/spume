# SPDX-License-Identifier: Apache-2.0
"""Writes a frame sequence: one compact binary glTF per frame, one spray.bin of float16 points, and meta.json.

Mesh frame: indices (uint16 or uint32), positions as int16 spread over the frame's own bounding box, normals as int8.
The accessors carry the bounds in `extras` so a reader can restore metres; it is a project format, not a general-purpose glTF.
Spray: per frame a run of float16 x y z vx vy vz (metres, metres per second); `meta.json` lists [first point, count] per frame.
Beside it `spray_ids.bin` (uint32 per point, stable for the point's life) and `spray_kinds.bin` (uint8 per point: 0 spray, 1 foam, 2 bubble).
A reader must treat `spray_kinds.bin` as optional: caches written before version 0.2 do not have it.
"""

import json
import struct
from pathlib import Path

import numpy as np

UNIT = "metres, +Y up, origin at the centre of the opening on the ground"


def _padded(data: bytes) -> bytes:
    return data + b"\x00" * (-len(data) % 4)


def encode_mesh_frame(vertices: np.ndarray, normals: np.ndarray, indices: np.ndarray) -> tuple[bytes, np.ndarray, np.ndarray]:
    """Returns (glb bytes, bounds minimum, bounds maximum)."""
    if len(vertices) == 0:
        vertices = np.zeros((3, 3), np.float32)
        normals = np.tile(np.array([[0, 1, 0]], np.float32), (3, 1))
        indices = np.array([0, 1, 2], np.uint32)
    bounds_min = vertices.min(axis=0)
    bounds_max = vertices.max(axis=0)
    span = np.maximum(bounds_max - bounds_min, 1.0e-6)
    quantised = np.round((vertices - bounds_min) / span * 65535.0 - 32768.0).astype(np.int16)
    quantised_normals = np.round(np.clip(normals, -1.0, 1.0) * 127.0).astype(np.int8)
    index_type = np.uint16 if len(vertices) < 65536 else np.uint32
    index_bytes = _padded(indices.astype(index_type).tobytes())
    position_bytes = _padded(quantised.tobytes())
    normal_bytes = _padded(quantised_normals.tobytes())
    description = {
        "asset": {"version": "2.0", "generator": "spume"},
        "buffers": [{"byteLength": len(index_bytes) + len(position_bytes) + len(normal_bytes)}],
        "bufferViews": [
            {"buffer": 0, "byteOffset": 0, "byteLength": len(index_bytes)},
            {"buffer": 0, "byteOffset": len(index_bytes), "byteLength": len(position_bytes)},
            {"buffer": 0, "byteOffset": len(index_bytes) + len(position_bytes), "byteLength": len(normal_bytes)},
        ],
        "accessors": [
            {"bufferView": 0, "componentType": 5123 if index_type is np.uint16 else 5125, "count": int(len(indices)), "type": "SCALAR"},
            {
                "bufferView": 1,
                "componentType": 5122,
                "count": int(len(vertices)),
                "type": "VEC3",
                "min": bounds_min.tolist(),
                "max": bounds_max.tolist(),
                "extras": {"quantised": "int16 over min..max"},
            },
            {"bufferView": 2, "componentType": 5120, "normalized": True, "count": int(len(vertices)), "type": "VEC3"},
        ],
        "meshes": [{"primitives": [{"attributes": {"POSITION": 1, "NORMAL": 2}, "indices": 0}]}],
        "nodes": [{"mesh": 0}],
        "scenes": [{"nodes": [0]}],
        "scene": 0,
    }
    json_bytes = json.dumps(description, separators=(",", ":")).encode()
    json_bytes += b" " * (-len(json_bytes) % 4)
    binary = index_bytes + position_bytes + normal_bytes
    total = 12 + 8 + len(json_bytes) + 8 + len(binary)
    glb = struct.pack("<III", 0x46546C67, 2, total)
    glb += struct.pack("<II", len(json_bytes), 0x4E4F534A) + json_bytes
    glb += struct.pack("<II", len(binary), 0x004E4942) + binary
    return glb, bounds_min, bounds_max


class SequenceWriter:
    def __init__(self, folder: Path, name: str, fps: int):
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)
        self.name = name
        self.fps = fps
        self.frame_count = 0
        self.bounds_min = np.full(3, np.inf)
        self.bounds_max = np.full(3, -np.inf)
        self.head_heights: list[float] = []
        self.spray_index: list[list[int]] = []
        self.spray_chunks: list[bytes] = []
        self.spray_id_chunks: list[bytes] = []
        self.spray_kind_chunks: list[bytes] = []
        self.spray_points = 0
        self.mesh_bytes = 0

    def add_frame(self, vertices, normals, indices, spray_positions, spray_velocities, spray_ids, head_height, spray_kinds=None):
        glb, low, high = encode_mesh_frame(vertices, normals, indices)
        (self.folder / f"frame_{self.frame_count:04d}.glb").write_bytes(glb)
        self.mesh_bytes += len(glb)
        if len(vertices):
            self.bounds_min = np.minimum(self.bounds_min, low)
            self.bounds_max = np.maximum(self.bounds_max, high)
        points = np.concatenate([spray_positions, spray_velocities], axis=1).astype(np.float16)
        self.spray_index.append([self.spray_points, int(len(points))])
        self.spray_chunks.append(points.tobytes())
        # each point keeps its own id for its whole life, so a renderer can give it the same size and jitter in every frame
        self.spray_id_chunks.append((np.asarray(spray_ids).astype(np.int64) & 0xFFFFFFFF).astype(np.uint32).tobytes())
        kinds = np.zeros(len(points), np.uint8) if spray_kinds is None else np.asarray(spray_kinds, np.uint8)
        self.spray_kind_chunks.append(kinds.tobytes())
        self.spray_points += len(points)
        self.head_heights.append(round(float(head_height), 3))
        self.frame_count += 1

    def finish(self, settings: dict, origin=None) -> dict:
        """`origin` is the point (solver metres) subtracted from every exported position; a reader adds it back to get box coordinates."""
        spray_bytes = b"".join(self.spray_chunks)
        (self.folder / "spray.bin").write_bytes(spray_bytes)
        spray_id_bytes = b"".join(self.spray_id_chunks)
        (self.folder / "spray_ids.bin").write_bytes(spray_id_bytes)
        spray_kind_bytes = b"".join(self.spray_kind_chunks)
        (self.folder / "spray_kinds.bin").write_bytes(spray_kind_bytes)
        meta = {
            "name": self.name,
            "fps": self.fps,
            "frameCount": self.frame_count,
            "frameFile": "frame_%04d.glb",
            "unit": UNIT,
            "meshEncoding": "int16-positions-int8-normals",
            "boundsMin": [round(float(value), 3) for value in self.bounds_min],
            "boundsMax": [round(float(value), 3) for value in self.bounds_max],
            "headHeights": self.head_heights,
            "spray": {"file": "spray.bin", "floatsPerPoint": 6, "dtype": "float16", "ids": "spray_ids.bin", "kinds": "spray_kinds.bin", "index": self.spray_index},
            "origin": None if origin is None else [round(float(value), 5) for value in origin],
            "settings": settings,
            "bytes": self.mesh_bytes + len(spray_bytes) + len(spray_id_bytes) + len(spray_kind_bytes),
        }
        (self.folder / "meta.json").write_text(json.dumps(meta, separators=(",", ":")))
        return meta

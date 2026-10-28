# SPDX-License-Identifier: Apache-2.0
"""Reads a Spume cache folder (frame_NNNN.glb, spray.bin, spray_ids.bin, spray_kinds.bin, meta.json) into numpy arrays."""

import json
import struct
from pathlib import Path

import numpy as np


def read_meta(folder):
    return json.loads(Path(folder, "meta.json").read_text())


def read_mesh_frame(folder, frame):
    """Returns (vertices float32 Nx3 in solver metres, triangle indices int32 Mx3)."""
    data = Path(folder, f"frame_{frame:04d}.glb").read_bytes()
    json_length = struct.unpack_from("<I", data, 12)[0]
    description = json.loads(data[20 : 20 + json_length])
    binary_start = 20 + json_length + 8
    views = description["bufferViews"]
    accessors = description["accessors"]

    def view_bytes(index):
        view = views[index]
        return data[binary_start + view["byteOffset"] : binary_start + view["byteOffset"] + view["byteLength"]]

    index_type = np.uint16 if accessors[0]["componentType"] == 5123 else np.uint32
    indices = np.frombuffer(view_bytes(0), index_type, count=accessors[0]["count"]).astype(np.int32)
    count = accessors[1]["count"]
    quantised = np.frombuffer(view_bytes(1), np.int16, count=count * 3).reshape(count, 3).astype(np.float64)
    low = np.array(accessors[1]["min"])
    high = np.array(accessors[1]["max"])
    span = np.maximum(high - low, 1.0e-6)
    vertices = ((quantised + 32768.0) / 65535.0 * span + low).astype(np.float32)
    return vertices, indices.reshape(-1, 3)


def read_spray_frame(folder, meta, frame):
    """Returns (positions Nx3, velocities Nx3, ids uint32 N, kinds uint8 N or None) in solver metres.
    Kinds are 0 spray, 1 foam, 2 bubble; caches from before version 0.2 have no kinds file and give None."""
    first, count = meta["spray"]["index"][frame]
    points = np.fromfile(Path(folder, meta["spray"]["file"]), np.float16, count=count * 6, offset=first * 12)
    points = points.reshape(count, 6).astype(np.float32)
    ids = np.fromfile(Path(folder, meta["spray"]["ids"]), np.uint32, count=count, offset=first * 4)
    kinds = None
    kinds_name = meta["spray"].get("kinds")
    if kinds_name and Path(folder, kinds_name).exists():
        kinds = np.fromfile(Path(folder, kinds_name), np.uint8, count=count, offset=first)
    return points[:, :3], points[:, 3:], ids, kinds

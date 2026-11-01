# SPDX-License-Identifier: Apache-2.0
"""Recorded kernel launches. `wp.launch` packs its arguments in Python, about 130 microseconds a call on an RTX 3060 against 20 for a recorded one,
and a solver step makes well over a hundred calls: most of a bake's time went there, not into the GPU."""

import warp as wp


class Launcher:
    """Keeps one recorded launch per kernel and set of arrays, and resends only the scalars that changed."""

    def __init__(self, device):
        self.device = device
        self.commands = {}

    def launch(self, kernel, dim, inputs):
        if dim == 0:
            return
        key = (kernel, *[id(value) for value in inputs if isinstance(value, wp.array)])
        entry = self.commands.get(key)
        if entry is None:
            command = wp.launch(kernel, dim=dim, inputs=inputs, device=self.device, record_cmd=True)
            self.commands[key] = (command, list(inputs))
            command.launch()
            return
        command, previous = entry
        for index, value in enumerate(inputs):
            if not isinstance(value, wp.array) and value != previous[index]:
                command.set_param_at_index(index, value)
                previous[index] = value
        command.set_dim(dim)
        command.launch()

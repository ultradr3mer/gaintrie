from enum import StrEnum
from typing import Any, NamedTuple

import numpy as np

from clarautils import symbol_to_str


class Char(StrEnum):
    fill = "."
    down = "↧"
    branch = "X"
    space = " "


def get_diff(string_a, sting_b):
    return "".join([Char.down if a != b
                    else Char.space
                    for a, b in zip(string_a, sting_b)])


def entropy_base_10_str(val):
    ceiling = np.array(np.ceil(val * 9), dtype=np.int32)
    ceiling = np.min((ceiling, np.full_like(ceiling, fill_value=9)), axis=0)
    return symbol_to_str(ceiling)


def entropy_before_str(entropy):
    return entropy_base_10_str(entropy.entropy_before)


def entropy_after_str(entropy):
    result = entropy_base_10_str(entropy.entropy_after)
    return "".join([(Char.space if i == entropy.idx else '') + c for i, c in enumerate(result)])


def entropy_gain_sum(entropy):
    return sum(entropy.entropy_before) - sum(entropy.entropy_after)


class RootBegin(NamedTuple):
    value: str


class NodeBegin(NamedTuple):
    name: str
    value_in: str
    value: str
    entropy: Any


class Strait(NamedTuple):
    op: Any
    value_before: str
    value_after: str


class NodeSplit(NamedTuple):
    bit_idx: int
    value_before: str
    value_after: str


class RootSplit(NamedTuple):
    value_before: str
    value_after: str
    gain: Any


class NodeEnd(NamedTuple):
    name: str


class Leaf(NamedTuple):
    name: str
    parent_value: str
    value: str
    full: str
    leaf_value: Any
    leaf_str: str


class TreePrinter:
    def __init__(self, realtime=False):
        self.realtime = realtime
        self.lines = []
        self.depth = 0
        self._handlers = {
            RootBegin: self.on_root_begin,
            NodeBegin: self.on_node_begin,
            Strait: self.on_strait,
            NodeSplit: self.on_node_split,
            RootSplit: self.on_root_split,
            NodeEnd: self.on_node_end,
            Leaf: self.on_leaf,
        }

    def handle(self, event):
        self._handlers[type(event)](event)

    def print(self):
        print("\n".join(self.lines))

    def _add(self, text, extra=0):
        line = "  " * (self.depth + extra) + text
        self.lines.append(line)
        if self.realtime:
            print(line)

    def on_root_begin(self, e):
        self._add(f"(root) [{e.value}]")
        self.depth += 1

    def on_node_begin(self, e):
        before = entropy_before_str(e.entropy)
        after = entropy_after_str(e.entropy)
        self._add(f"({e.name}) in: {e.value_in}  node: {e.value}  "
                  f"entropy: {before} -> {after}  gain: {entropy_gain_sum(e.entropy):.3f}")
        self.depth += 1

    def on_strait(self, e):
        self._add(f"{e.op}: {get_diff(e.value_before, e.value_after)} now: {e.value_after}")

    def on_node_split(self, e):
        self._add(f"op:{e.bit_idx}={Char.branch}: {get_diff(e.value_before, e.value_after)} out: {e.value_after}")

    def on_root_split(self, e):
        self._add(f"split: {get_diff(e.value_before, e.value_after)} out: {e.value_after} gain: {e.gain}")

    def on_node_end(self, e):
        self.depth -= 1
        self._add(f"({e.name}) End")

    def on_leaf(self, e):
        self._add(f"({e.name}) in: {e.parent_value}  node: {e.value}  "
                  f"leaf: ({e.leaf_value})-[{e.leaf_str}]  out: {e.full}")

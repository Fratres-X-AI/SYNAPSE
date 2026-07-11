import importlib
import sys

import test_showcase
from synapse_launcher import SCRIPTS, SCRIPT_MODULES


def test_showcase_is_wired_in_launcher():
    assert SCRIPTS["showcase"] == "synapse_showcase.py"
    assert SCRIPT_MODULES["synapse_showcase.py"] == "test_showcase"


def test_showcase_module_imports():
    module = importlib.import_module("test_showcase")
    assert hasattr(module, "main")
    assert hasattr(module, "parse_args")


def test_showcase_parse_args_accepts_fullscreen(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["test_showcase", "--fullscreen"])
    args = test_showcase.parse_args()
    assert test_showcase._resolve_fullscreen(args) is True


def test_showcase_parse_args_defaults_fullscreen(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["test_showcase"])
    args = test_showcase.parse_args()
    assert test_showcase._resolve_fullscreen(args) is True


def test_showcase_parse_args_windowed(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["test_showcase", "--windowed"])
    args = test_showcase.parse_args()
    assert test_showcase._resolve_fullscreen(args) is False

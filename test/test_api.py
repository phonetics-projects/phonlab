"""Every name exported by phonlab must resolve to a callable."""
import ast
import warnings
from pathlib import Path

import pytest

import phonlab

warnings.filterwarnings("ignore")


def _stub_names():
    stub = Path(phonlab.__file__).with_suffix(".pyi")
    names = []
    for node in ast.walk(ast.parse(stub.read_text())):
        if isinstance(node, ast.ImportFrom):
            names += [a.name for a in node.names]
    return names


@pytest.mark.parametrize("name", _stub_names())
def test_export_resolves(name):
    assert callable(getattr(phonlab, name))


def test_all_matches_stub():
    assert sorted(phonlab.__all__) == sorted(_stub_names())

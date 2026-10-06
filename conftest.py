"""Make pytest use the source tree rather than any installed copy of phonlab,
and add the --update-golden option used by test/test_regression.py."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def pytest_addoption(parser):
    parser.addoption(
        "--update-golden",
        action="store_true",
        default=False,
        help="Rewrite the golden files in test/golden from current output "
             "instead of comparing against them.",
    )

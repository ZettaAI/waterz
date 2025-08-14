import sys
from pathlib import Path

# waterz is not locally importable;
# this snippet forces the tests to use the installed version.
# See here for more details:
# https://stackoverflow.com/questions/67176036/how-to-prevent-pytest-using-local-module
project_dir = Path(__file__).resolve().parent.parent
# Only remove the exact project root; keep virtualenv/site-packages paths under it
sys.path = [p for p in sys.path if Path(p).resolve() != project_dir]

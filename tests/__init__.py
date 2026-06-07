"""Test package bootstrap.

Adds the ``src/`` directory to ``sys.path`` so the tests can import
``llm_persona`` directly with the standard-library test runner::

    python3 -m unittest discover -s tests

without requiring an editable install. (When the package *is* installed,
the installed copy still takes precedence because it appears earlier on
``sys.path``.)
"""

from __future__ import annotations

import os
import sys

_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if os.path.isdir(_SRC) and _SRC not in sys.path:
    sys.path.append(_SRC)

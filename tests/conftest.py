"""Test-wide setup that has to happen before anything imports ``server``.

``server.config`` reads the environment once, at import, and ``server.db``
builds its engine from it at import too. A test module that sets those
variables at its own import time therefore only works when it is the first
module imported — which made the fleet tests pass on their own and fail in a
suite, depending on nothing more than a filename.

Setting them here fixes that, and does something more important besides: it
keeps the real ``server_state.db`` out of the test run. That file holds the
scheduler's measured SLA observations, which are evidence; a test that
happened to write to it would be quietly editing a result.
"""

from __future__ import annotations

import os
import tempfile

_TEST_DB = os.path.join(tempfile.mkdtemp(prefix="vs-tests-"), "server_state.db")
os.environ.setdefault("VS_DATABASE_URL", f"sqlite:///{_TEST_DB}")
os.environ["VS_DATABASE_URL"] = f"sqlite:///{_TEST_DB}"

# A two-class fleet, so routing tests have something to choose between. Any
# module that needs a different one overrides config.LOCAL_FLEET directly.
os.environ.setdefault(
    "VS_LOCAL_FLEET",
    '[{"class":"cpu-fast","speed":1.0},{"class":"cpu-slow","speed":0.1}]',
)

# Never let a test run send a real message, whatever else is configured.
os.environ["VS_SMTP_HOST"] = ""
os.environ["VS_SMS_PROVIDER"] = ""

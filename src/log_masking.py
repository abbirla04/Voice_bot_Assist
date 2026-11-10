"""Keep secrets out of log files.

Meta's webhook check sends the verify token in the URL (?hub.verify_token=...), and aiohttp's
access log writes the whole URL. This filter masks it before any handler sees the line."""

from __future__ import annotations

import logging
import re

_SECRET_PARAMS = re.compile(r"((?:hub\.verify_token|access_token)=)[^&\s\"]+")


class MaskSecrets(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        masked = _SECRET_PARAMS.sub(r"\1***", msg)
        if masked != msg:
            record.msg, record.args = masked, ()
        return True


def install() -> None:
    logging.getLogger("aiohttp.access").addFilter(MaskSecrets())

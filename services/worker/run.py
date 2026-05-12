from __future__ import annotations

import asyncio
import logging

from services.api.app.core.config import load_settings
from services.worker.app import ColorizationWorker


def main() -> None:
    settings = load_settings()
    logging.basicConfig(
        level=settings.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    worker = ColorizationWorker.from_settings(settings)
    asyncio.run(worker.run_forever())


if __name__ == "__main__":
    main()

import os
import asyncio

import uvloop


# Keep Celery queue calls local for tests (no Redis dependency).
os.environ.setdefault("CELERY_BROKER_URL", "memory://")
os.environ.setdefault("CELERY_RESULT_BACKEND", "cache+memory://")

# This environment blocks writes to the default asyncio selector wakeup socket.
# uvloop uses a different mechanism that keeps TestClient responsive.
asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

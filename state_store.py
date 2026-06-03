from __future__ import annotations

import os

# Set USER_STATE_BACKEND=postgres to use store_postgres (requires DATABASE_URL).
# Defaults to store_sqlite (zero setup).
_BACKEND = os.getenv("USER_STATE_BACKEND", "sqlite").lower()

if _BACKEND == "postgres":
    from store_postgres import (  # noqa: F401
        get_or_create_user,
        init_db,
        log_interaction,
        recent_interactions,
        set_level,
        set_stage,
    )
elif _BACKEND == "sqlite":
    from store_sqlite import (  # noqa: F401
        get_or_create_user,
        init_db,
        log_interaction,
        recent_interactions,
        set_level,
        set_stage,
    )
else:
    raise ValueError(
        f"unknown USER_STATE_BACKEND {_BACKEND!r}; use 'sqlite' or 'postgres'"
    )

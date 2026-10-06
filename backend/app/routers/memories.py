"""
Memories router — exposes the SQLite durable-memory store to the Flutter app.

GET  /memories            — all durable memories
GET  /memories/recent     — recent memories with optional ?limit= and ?category=
"""
from typing import Optional

from fastapi import APIRouter, Depends, Query

from app.routers.auth import current_actor
from app.memory.store import SQLiteMemoryStore

router = APIRouter(prefix="/memories", tags=["memories"])


@router.get("")
def list_all_memories(actor=Depends(current_actor)):
    store = SQLiteMemoryStore()
    return {"memories": store.list_all()}


@router.get("/recent")
def recent_memories(
    limit: int = Query(default=5, ge=1, le=50),
    category: Optional[str] = Query(default=None),
    actor=Depends(current_actor),
):
    store = SQLiteMemoryStore()
    memories = store.list_all()
    if category:
        memories = [m for m in memories if m.get("category") == category]
    # list_all returns newest-updated first; take the top `limit`
    return {"memories": memories[:limit]}

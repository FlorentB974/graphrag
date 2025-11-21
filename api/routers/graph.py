"""Graph data endpoints for visualization."""

from fastapi import APIRouter, HTTPException, Query

from core.graph_db import graph_db

router = APIRouter()


@router.get("/visualization")
def get_graph_visualization(min_relationship_strength: float = Query(0.0, ge=0.0)):
    """Return entity graph data suitable for interactive visualization."""
    try:
        return graph_db.get_graph_visualization_data(
            min_relationship_strength=min_relationship_strength
        )
    except Exception as exc:  # pragma: no cover - surfaced to caller
        raise HTTPException(status_code=500, detail=str(exc)) from exc

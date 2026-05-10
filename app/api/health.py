"""Health-check endpoint."""

from fastapi import APIRouter

from app.models.schemas import HealthResponse

router = APIRouter(prefix="/health", tags=["health"])


@router.get("", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Return the application health status."""
    return HealthResponse(status="ok")


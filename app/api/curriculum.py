"""Administrative curriculum ingestion endpoints."""

import hmac

from fastapi import APIRouter, Depends, Header, HTTPException, status

from app.config import Settings, get_settings
from app.database.vector_store import CurriculumVectorStore, VectorStoreError
from app.models.schemas import LessonCreate, LessonCreatedResponse

router = APIRouter(prefix="/curriculum", tags=["curriculum"])


def verify_admin_api_key(
    x_admin_api_key: str | None = Header(default=None, alias="X-Admin-API-Key"),
    settings: Settings = Depends(get_settings),
) -> None:
    """Protect curriculum admin routes with a constant-time API-key check."""
    if not settings.admin_api_key:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="admin authentication is not configured",
        )
    if x_admin_api_key is None or not hmac.compare_digest(
        x_admin_api_key,
        settings.admin_api_key,
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="invalid admin credentials",
        )


def get_curriculum_vector_store(settings: Settings = Depends(get_settings)) -> CurriculumVectorStore:
    """Create the curriculum vector store dependency."""
    return CurriculumVectorStore(settings=settings)


@router.post(
    "/lesson",
    response_model=LessonCreatedResponse,
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(verify_admin_api_key)],
)
async def create_lesson(
    payload: LessonCreate,
    vector_store: CurriculumVectorStore = Depends(get_curriculum_vector_store),
) -> LessonCreatedResponse:
    """Persist an English exercise and correction rubric for future RAG retrieval."""
    try:
        lesson_id = await vector_store.add_lesson(
            title=payload.title,
            content=payload.content,
        )
    except VectorStoreError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="curriculum storage is unavailable",
        ) from exc

    return LessonCreatedResponse(id=lesson_id, title=payload.title)

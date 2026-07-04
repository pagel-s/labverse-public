"""Paper bundle inspection routes."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from labverse.paper import (
    BundleChatViewReport,
    BundleRunSummary,
    BundleStatusReport,
    BundleStepDetail,
    BundleStepSummary,
    build_bundle_chat_view,
    bundle_status,
    get_bundle_step_detail,
    list_bundle_runs,
    list_bundle_steps,
)

router = APIRouter(prefix="/paper", tags=["paper"])


@router.get("/health")
async def paper_health() -> dict[str, str]:
    return {"status": "ok", "service": "paper"}


@router.get("/status", response_model=BundleStatusReport)
async def paper_status(
    bundle_dir: str = Query(..., description="Path to the paper bundle directory."),
) -> BundleStatusReport:
    try:
        return bundle_status(bundle_dir)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/runs", response_model=list[BundleRunSummary])
async def paper_runs(
    bundle_dir: str = Query(..., description="Path to the paper bundle directory."),
) -> list[BundleRunSummary]:
    try:
        return list_bundle_runs(bundle_dir)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/chat-view", response_model=BundleChatViewReport)
async def paper_chat_view(
    bundle_dir: str = Query(..., description="Path to the paper bundle directory."),
    run: str | None = Query(
        None,
        description="Optional relative run path under the bundle.",
    ),
    step: int | None = Query(
        None,
        ge=1,
        description="Optional 1-based step index to render.",
    ),
    tail: int = Query(
        1,
        ge=1,
        description="Number of latest steps to render when step is unset.",
    ),
    show_tool_schema: bool = Query(
        False,
        description="Include the full action schema JSON for each rendered step.",
    ),
) -> BundleChatViewReport:
    try:
        return build_bundle_chat_view(
            bundle_dir,
            run=run,
            step=step,
            tail=tail,
            show_tool_schema=show_tool_schema,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/steps", response_model=list[BundleStepSummary])
async def paper_steps(
    bundle_dir: str = Query(..., description="Path to the paper bundle directory."),
    run: str | None = Query(
        None,
        description="Optional relative run path under the bundle.",
    ),
) -> list[BundleStepSummary]:
    try:
        return list_bundle_steps(bundle_dir, run=run)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/step", response_model=BundleStepDetail)
async def paper_step(
    bundle_dir: str = Query(..., description="Path to the paper bundle directory."),
    run: str = Query(..., description="Relative run path under the bundle."),
    step: int = Query(..., ge=1, description="1-based step index."),
) -> BundleStepDetail:
    try:
        return get_bundle_step_detail(bundle_dir, run=run, step=step)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


__all__ = ["router"]

"""상태/스키마 모델 정의."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Annotated, Any

from pydantic import BaseModel, Field, field_validator
from typing_extensions import TypedDict


def merge_unique_list(left: list[Any] | None, right: list[Any] | None) -> list[Any]:
    """리스트를 순서를 유지하며 중복 제거 병합한다."""
    left = left or []
    right = right or []
    merged: list[Any] = []
    seen: set[str] = set()

    for item in [*left, *right]:
        token = json.dumps(item, ensure_ascii=False, sort_keys=True, default=str)
        if token not in seen:
            seen.add(token)
            merged.append(item)
    return merged


def merge_dict(left: dict[str, Any] | None, right: dict[str, Any] | None) -> dict[str, Any]:
    """우측 값을 우선 적용하는 딕셔너리 병합."""
    merged: dict[str, Any] = dict(left or {})
    merged.update(right or {})
    return merged


def merge_dict_of_lists(
    left: dict[str, list[Any]] | None,
    right: dict[str, list[Any]] | None,
) -> dict[str, list[Any]]:
    """키별 리스트를 병합하며 중복을 제거한다."""
    left = left or {}
    right = right or {}
    merged: dict[str, list[Any]] = {key: list(value) for key, value in left.items()}

    for key, values in right.items():
        existing = merged.setdefault(key, [])
        merged[key] = merge_unique_list(existing, values)

    return merged


class ParsedRequest(BaseModel):
    """사용자 요청 파싱 결과."""

    target_technologies: list[str] = Field(default_factory=lambda: ["HBM4", "PIM", "CXL"])
    target_competitors: list[str] = Field(default_factory=lambda: ["Samsung", "Micron", "SK hynix"])
    analysis_focus: str = Field(default="기술 성숙도와 위협 수준 비교")


class SourceItem(BaseModel):
    """Technology Scanner 결과 단위."""

    technology: str
    company: str = Field(default="Unknown")
    source_group: str
    query: str
    title: str
    url: str
    published_date: str = Field(default="")
    summary: str


class IngestedPaper(BaseModel):
    """논문 수집/색인 메타데이터."""

    url: str
    pdf_url: str
    title: str
    num_chunks: int = 0
    status: str = "success"
    error_message: str = ""


class IndirectIndicators(BaseModel):
    """간접 지표 분석 결과."""

    patent_trend: str = "unknown"
    publication_frequency: str = "unknown"
    hiring_keywords: list[str] = Field(default_factory=list)
    supply_agreements: list[str] = Field(default_factory=list)
    estimated_investment_level: str = "unknown"


class CompetitorProfile(BaseModel):
    """경쟁사-기술 조합 프로파일."""

    company: str
    technology: str
    product_portfolio: list[str] = Field(default_factory=list)
    technical_differentiators: list[str] = Field(default_factory=list)
    partnerships: list[str] = Field(default_factory=list)
    indirect_indicators: IndirectIndicators = Field(default_factory=IndirectIndicators)
    public_roadmap: list[str] = Field(default_factory=list)
    profile_confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    key_sources: list[str] = Field(default_factory=list)


class TRLAssessment(BaseModel):
    """TRL 평가 결과."""

    technology: str
    competitor: str
    trl_level: int = Field(ge=1, le=9)
    trl_sublevel: str = Field(default="mid")
    confidence: float = Field(ge=0.0, le=1.0)
    assessment_basis: str = Field(default="direct")
    evidence: list[str] = Field(default_factory=list)
    indirect_indicators_used: list[str] = Field(default_factory=list)
    information_gaps: list[str] = Field(default_factory=list)
    assessment_note: str = Field(default="")

    @field_validator("assessment_basis")
    @classmethod
    def validate_basis(cls, value: str) -> str:
        normalized = value.lower().strip()
        if normalized not in {"direct", "indirect"}:
            raise ValueError("assessment_basis는 'direct' 또는 'indirect'만 허용됩니다.")
        return normalized


class ThreatAssessment(BaseModel):
    """경쟁 위협 평가 결과."""

    competitor: str
    technology: str
    threat_level: int = Field(ge=1, le=5)
    threat_factors: list[str] = Field(default_factory=list)
    mitigating_factors: list[str] = Field(default_factory=list)
    time_horizon: str = Field(default="unknown")
    strategic_recommendation: str = Field(default="")


class ReportEvaluation(BaseModel):
    """최종 보고서 검증 결과."""

    purpose_focus: bool
    competitor_comparison: bool
    integrated_implications: bool
    reliability_metrics_reported: bool
    summary_actionable: bool
    structural_completeness: bool
    evidence_citation: bool
    citation_format_compliance: bool
    logical_coherence: bool
    gap_acknowledgment: bool
    overall_pass: bool
    feedback: list[str] = Field(default_factory=list)


class AssessmentQuality(BaseModel):
    """TRL 평가 단계 품질 게이트 결과."""

    coverage_complete: bool
    evidence_sufficient: bool
    consistency_ok: bool
    overall_pass: bool
    feedback: list[str] = Field(default_factory=list)


class AppConfig(BaseModel):
    """실행 구성."""

    base_dir: Path
    data_dir: Path
    outputs_dir: Path
    prompts_dir: Path
    openai_model: str = "gpt-4o-mini"
    openai_temperature: float = 0.1
    tavily_max_results: int = 8
    tavily_lookback_days: int = 1095
    top_k_retrieval: int = 5
    max_iterations: int = 2
    embedding_provider: str = "huggingface"  # "huggingface" | "jina" | "voyage"
    embedding_model: str = "BAAI/bge-m3"
    fallback_embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    jina_api_key: str = ""
    voyage_api_key: str = ""
    qdrant_collection_name: str = "tech_strategy_docs"
    recursion_limit: int = 60

    @classmethod
    def from_env(cls, base_dir: Path) -> "AppConfig":
        """환경 변수 기반으로 기본 설정을 만든다."""
        return cls(
            base_dir=base_dir,
            data_dir=base_dir / "data",
            outputs_dir=base_dir / "outputs",
            prompts_dir=base_dir / "prompts",
            openai_model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            openai_temperature=float(os.getenv("OPENAI_TEMPERATURE", "0.1")),
            tavily_max_results=int(os.getenv("TAVILY_MAX_RESULTS", "8")),
            tavily_lookback_days=int(os.getenv("TAVILY_LOOKBACK_DAYS", "1095")),
            top_k_retrieval=int(os.getenv("TOP_K_RETRIEVAL", "5")),
            max_iterations=int(os.getenv("MAX_ITERATIONS", "2")),
            embedding_provider=os.getenv("EMBEDDING_PROVIDER", "huggingface"),
            embedding_model=os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3"),
            fallback_embedding_model=os.getenv(
                "FALLBACK_EMBEDDING_MODEL",
                "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            ),
            jina_api_key=os.getenv("JINA_API_KEY", ""),
            voyage_api_key=os.getenv("VOYAGE_API_KEY", ""),
            qdrant_collection_name=os.getenv("QDRANT_COLLECTION_NAME", "tech_strategy_docs"),
            recursion_limit=int(os.getenv("GRAPH_RECURSION_LIMIT", "60")),
        )


class TechStrategyState(TypedDict, total=False):
    """LangGraph 전역 상태."""

    topic: str
    user_request: str
    phase: str
    target_technologies: list[str]
    target_competitors: list[str]
    collected_data: Annotated[dict[str, list[dict[str, Any]]], merge_dict_of_lists]
    paper_urls: Annotated[list[str], merge_unique_list]
    ingested_papers: Annotated[list[dict[str, Any]], merge_unique_list]
    domain_context: Annotated[dict[str, list[str]], merge_dict_of_lists]
    competitor_profiles: Annotated[dict[str, dict[str, Any]], merge_dict]
    trl_assessments: Annotated[list[dict[str, Any]], merge_unique_list]
    threat_analysis: Annotated[list[dict[str, Any]], merge_unique_list]
    references: Annotated[list[str], merge_unique_list]
    reference_items: Annotated[list[dict[str, Any]], merge_unique_list]
    report_draft: str
    quality_scores: Annotated[dict[str, Any], merge_dict]
    quality_gate: str
    review_decision: str
    revision_feedback: list[str]
    errors: Annotated[list[str], merge_unique_list]
    run_dir: str
    report_path: str
    next_node: str
    iteration_count: int
    max_iterations: int


def build_initial_state(topic: str, user_request: str, max_iterations: int) -> TechStrategyState:
    """그래프 실행 초기 상태."""
    return TechStrategyState(
        topic=topic,
        user_request=user_request,
        phase="init",
        target_technologies=["HBM4", "PIM", "CXL"],
        target_competitors=["Samsung", "Micron", "SK hynix"],
        collected_data={},
        paper_urls=[],
        ingested_papers=[],
        domain_context={},
        competitor_profiles={},
        trl_assessments=[],
        threat_analysis=[],
        references=[],
        reference_items=[],
        report_draft="",
        quality_scores={},
        quality_gate="unknown",
        review_decision="unknown",
        revision_feedback=[],
        errors=[],
        run_dir="",
        report_path="",
        next_node="",
        iteration_count=0,
        max_iterations=max_iterations,
    )

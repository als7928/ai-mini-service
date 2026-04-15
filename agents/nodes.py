"""LangGraph 노드(에이전트) 정의."""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from langchain_core.documents import Document

from .decorators import ensure_state_keys, log_execution_time, node_exception_handler
from .models import (
    AssessmentQuality,
    AppConfig,
    CompetitorProfile,
    ParsedRequest,
    ReportEvaluation,
    ThreatAssessment,
    TRLAssessment,
)
from .prompts import PromptRepository
from .services import OutputService, PaperIngestionService, TavilySearchService, VectorStoreService


class BaseNode:
    """노드 공통 의존성."""

    def __init__(
        self,
        config: AppConfig,
        prompts: PromptRepository,
        llm: Any,
        logger: logging.Logger,
    ) -> None:
        self.config = config
        self.prompts = prompts
        self.llm = llm
        self.logger = logger


class RequestParserNode(BaseNode):
    """요청 파싱 노드."""

    @node_exception_handler(node_name="request_parser", fallback_goto="save_output")
    @log_execution_time("request_parser")
    @ensure_state_keys(["topic", "user_request"])
    def run(self, state: dict[str, Any]) -> dict[str, Any]:
        prompt = self.prompts.render(
            "request_parser_system.txt",
            topic=state.get("topic", ""),
            user_request=state.get("user_request", ""),
        )

        parsed = self.llm.with_structured_output(ParsedRequest).invoke(
            [
                ("system", prompt),
                (
                    "human",
                    "요청을 파싱해 target_technologies / target_competitors / analysis_focus를 생성해라.",
                ),
            ]
        )

        return {
            "phase": "request_parsed",
            "target_technologies": parsed.target_technologies or state["target_technologies"],
            "target_competitors": parsed.target_competitors or state["target_competitors"],
        }


class TechnologyScannerNode(BaseNode):
    """웹 정보 수집 노드."""

    def __init__(
        self,
        config: AppConfig,
        prompts: PromptRepository,
        llm: Any,
        logger: logging.Logger,
        search_service: TavilySearchService,
    ) -> None:
        super().__init__(config=config, prompts=prompts, llm=llm, logger=logger)
        self.search_service = search_service

    @node_exception_handler(node_name="technology_scanner", fallback_goto="save_output")
    @log_execution_time("technology_scanner")
    @ensure_state_keys(["target_technologies", "target_competitors", "topic"])
    def run(self, state: dict[str, Any]) -> dict[str, Any]:
        collected_data: dict[str, list[dict[str, Any]]] = {}
        all_sources = []

        for technology in state["target_technologies"]:
            found = self.search_service.search_by_technology(
                technology=technology,
                competitors=state["target_competitors"],
                topic=state["topic"],
            )
            collected_data[technology] = [item.model_dump() for item in found]
            all_sources.extend(found)

        paper_urls = self.search_service.extract_arxiv_urls(all_sources)
        references = [item.url for item in all_sources if item.url]
        reference_items = [
            {
                "title": item.title,
                "url": item.url,
                "published_date": item.published_date,
                "company": item.company,
                "source_group": item.source_group,
            }
            for item in all_sources
            if item.url
        ]

        return {
            "phase": "scanned",
            "collected_data": collected_data,
            "paper_urls": paper_urls,
            "references": references,
            "reference_items": reference_items,
        }


class PaperIngestorNode(BaseNode):
    """논문 PDF 수집 + 벡터스토어 색인 노드."""

    def __init__(
        self,
        config: AppConfig,
        prompts: PromptRepository,
        llm: Any,
        logger: logging.Logger,
        ingestion_service: PaperIngestionService,
        vector_service: VectorStoreService,
    ) -> None:
        super().__init__(config=config, prompts=prompts, llm=llm, logger=logger)
        self.ingestion_service = ingestion_service
        self.vector_service = vector_service

    @node_exception_handler(node_name="paper_ingestor", fallback_goto="save_output")
    @log_execution_time("paper_ingestor")
    @ensure_state_keys(["paper_urls", "collected_data"])
    def run(self, state: dict[str, Any]) -> dict[str, Any]:
        arxiv_urls = list(state.get("paper_urls", []))
        ingested, paper_docs = self.ingestion_service.ingest_arxiv_urls(
            arxiv_urls=arxiv_urls,
            data_dir=self.config.data_dir / "papers",
        )

        # 웹 수집 요약도 함께 벡터스토어에 올려 RAG 커버리지를 보강한다.
        web_docs = self._build_web_documents(state.get("collected_data", {}))
        all_docs = [*paper_docs, *web_docs]

        chunk_count = 0
        retrieval_metrics = {
            "retriever": "qdrant_similarity",
            "num_queries": 0,
            "Hit@1": 0.0,
            "Hit@3": 0.0,
            "Hit@5": 0.0,
            "MRR": 0.0,
        }
        if all_docs:
            chunk_count = self.vector_service.build(all_docs)
            retrieval_metrics = self.vector_service.compute_retrieval_metrics(sample_size=20)

        paper_reference_items = [
            {
                "title": paper.title,
                "url": paper.url,
                "published_date": "",
                "company": "Unknown",
                "source_group": "academic",
            }
            for paper in ingested
            if paper.url
        ]

        return {
            "phase": "papers_ingested",
            "ingested_papers": [item.model_dump() for item in ingested],
            "references": [paper.url for paper in ingested if paper.url],
            "reference_items": paper_reference_items,
            "quality_scores": {
                "indexed_chunks": chunk_count,
                "retrieval_metrics": retrieval_metrics,
            },
        }

    def _build_web_documents(self, collected_data: dict[str, list[dict[str, Any]]]) -> list[Document]:
        docs: list[Document] = []
        for technology, items in collected_data.items():
            for item in items:
                summary = str(item.get("summary", "")).strip()
                url = str(item.get("url", "")).strip()
                title = str(item.get("title", "")).strip()
                if not summary:
                    continue
                docs.append(
                    Document(
                        page_content=f"[{technology}] {title}\n{summary}",
                        metadata={"source": url, "technology": technology, "title": title},
                    )
                )
        return docs


class DomainKnowledgeNode(BaseNode):
    """RAG 기반 도메인 지식 보강 노드."""

    def __init__(
        self,
        config: AppConfig,
        prompts: PromptRepository,
        llm: Any,
        logger: logging.Logger,
        vector_service: VectorStoreService,
    ) -> None:
        super().__init__(config=config, prompts=prompts, llm=llm, logger=logger)
        self.vector_service = vector_service

    @node_exception_handler(node_name="domain_knowledge", fallback_goto="save_output")
    @log_execution_time("domain_knowledge")
    @ensure_state_keys(["target_technologies"])
    def run(self, state: dict[str, Any]) -> dict[str, Any]:
        domain_context: dict[str, list[str]] = {}
        rag_sources: list[str] = []
        key_terms = [*state["target_technologies"], "TRL framework", "HBM bandwidth", "PIM energy efficiency", "CXL"]

        for term in key_terms:
            docs = self.vector_service.retrieve(query=term, k=self.config.top_k_retrieval)
            snippets = [doc.page_content[:500] for doc in docs if doc.page_content]
            for doc in docs:
                source = str(getattr(doc, "metadata", {}).get("source", "")).strip()
                if source.startswith("http"):
                    rag_sources.append(source)

            # 벡터 검색 결과가 없으면 LLM으로 최소 배경 설명을 생성한다.
            if not snippets:
                fallback = self.llm.invoke(
                    [
                        ("system", self.prompts.read("domain_knowledge_system.txt")),
                        ("human", f"{term}의 핵심 개념을 3문장 이내로 설명하라."),
                    ]
                ).content
                snippets = [str(fallback).strip()]
            domain_context[term] = snippets

        return {"phase": "domain_enriched", "domain_context": domain_context, "rag_sources": rag_sources}


class CompetitorProfilerNode(BaseNode):
    """경쟁사 프로파일 생성 노드."""

    @node_exception_handler(node_name="competitor_profiler", fallback_goto="save_output")
    @log_execution_time("competitor_profiler")
    @ensure_state_keys(["target_competitors", "target_technologies", "collected_data", "domain_context"])
    def run(self, state: dict[str, Any]) -> dict[str, Any]:
        profiles: dict[str, dict[str, Any]] = {}
        system_prompt = self.prompts.read("competitor_profiler_system.txt")

        for competitor in state["target_competitors"]:
            for technology in state["target_technologies"]:
                key = f"{competitor}_{technology}"
                related_items = self._filter_related_items(state["collected_data"], competitor, technology)
                context = state["domain_context"].get(technology, [])

                profile = self.llm.with_structured_output(CompetitorProfile).invoke(
                    [
                        ("system", system_prompt),
                        (
                            "human",
                            self._build_profile_prompt(
                                competitor=competitor,
                                technology=technology,
                                related_items=related_items,
                                context=context,
                            ),
                        ),
                    ]
                )
                profile_data = profile.model_dump()
                if not profile_data.get("key_sources"):
                    profile_data["key_sources"] = self._extract_source_urls(related_items)[:8]
                profiles[key] = profile_data

        return {"phase": "profiled", "competitor_profiles": profiles}

    def _extract_source_urls(self, related_items: list[dict[str, Any]]) -> list[str]:
        urls: list[str] = []
        seen: set[str] = set()
        for item in related_items:
            url = str(item.get("url", "")).strip()
            if not url or url in seen:
                continue
            seen.add(url)
            urls.append(url)
        return urls

    def _filter_related_items(
        self,
        collected_data: dict[str, list[dict[str, Any]]],
        competitor: str,
        technology: str,
    ) -> list[dict[str, Any]]:
        items = collected_data.get(technology, [])
        return [item for item in items if competitor.lower() in json.dumps(item, ensure_ascii=False).lower()]

    def _build_profile_prompt(
        self,
        competitor: str,
        technology: str,
        related_items: list[dict[str, Any]],
        context: list[str],
    ) -> str:
        return (
            f"[Competitor]\n{competitor}\n\n"
            f"[Technology]\n{technology}\n\n"
            f"[Collected Sources]\n{json.dumps(related_items[:10], ensure_ascii=False, indent=2)}\n\n"
            f"[Domain Context]\n{json.dumps(context[:5], ensure_ascii=False, indent=2)}\n\n"
            "공개 근거만 활용해 구조화된 프로파일을 작성하라."
        )


class TRLAssessorNode(BaseNode):
    """TRL/위협 분석 노드."""

    @node_exception_handler(node_name="trl_assessor", fallback_goto="save_output")
    @log_execution_time("trl_assessor")
    @ensure_state_keys(["competitor_profiles", "domain_context"])
    def run(self, state: dict[str, Any]) -> dict[str, Any]:
        assessments: list[dict[str, Any]] = []
        threats: list[dict[str, Any]] = []
        system_prompt = self.prompts.read("trl_assessor_system.txt")

        for profile_key, profile in state["competitor_profiles"].items():
            assessment = self.llm.with_structured_output(TRLAssessment).invoke(
                [
                    ("system", system_prompt),
                    (
                        "human",
                        self._build_trl_prompt(profile=profile, domain_context=state["domain_context"]),
                    ),
                ]
            )
            threat = self.llm.with_structured_output(ThreatAssessment).invoke(
                [
                    ("system", "너는 반도체 기술 위협 평가 전문가다."),
                    ("human", self._build_threat_prompt(profile=profile, assessment=assessment.model_dump())),
                ]
            )

            assessments.append(assessment.model_dump())
            threats.append(threat.model_dump())
            self.logger.info("TRL 평가 완료: %s", profile_key)

        return {
            "phase": "assessed",
            "trl_assessments": assessments,
            "threat_analysis": threats,
        }

    def _build_trl_prompt(self, profile: dict[str, Any], domain_context: dict[str, list[str]]) -> str:
        return (
            f"[Profile]\n{json.dumps(profile, ensure_ascii=False, indent=2)}\n\n"
            f"[TRL Context]\n{json.dumps(domain_context.get('TRL framework', []), ensure_ascii=False, indent=2)}\n\n"
            "TRL 1~9 중 하나를 선택하고, 4~6이면 assessment_basis='indirect'로 표시하라.\n"
            "추가 규칙:\n"
            "- evidence는 최소 2개 이상 작성하라. 각 항목은 가능한 한 구체적인 관찰/사실 + 출처 URL을 포함하라.\n"
            "- assessment_note에 '왜 그 TRL인지'를 2~3문장으로 요약하라.\n"
            "- information_gaps는 최소 1개 이상 작성하라."
        )

    def _build_threat_prompt(self, profile: dict[str, Any], assessment: dict[str, Any]) -> str:
        return (
            f"[Profile]\n{json.dumps(profile, ensure_ascii=False, indent=2)}\n\n"
            f"[Assessment]\n{json.dumps(assessment, ensure_ascii=False, indent=2)}\n\n"
            "위협 수준(1~5)을 평가하고 대응 전략을 제시하라."
        )


class QualityCheckNode(BaseNode):
    """TRL/프로파일링 품질 게이트 노드."""

    @node_exception_handler(node_name="quality_check", fallback_goto="save_output")
    @log_execution_time("quality_check")
    @ensure_state_keys(["trl_assessments", "target_technologies", "target_competitors", "iteration_count", "max_iterations"])
    def run(self, state: dict[str, Any]) -> dict[str, Any]:
        expected_count = len(state["target_technologies"]) * len(state["target_competitors"])
        assessments = state.get("trl_assessments", [])

        evidence_counts = [len(item.get("evidence", [])) for item in assessments if isinstance(item, dict)]
        avg_evidence = (sum(evidence_counts) / len(evidence_counts)) if evidence_counts else 0.0

        basic_pass = len(assessments) >= max(1, expected_count) and avg_evidence >= 1.0
        fallback_consistency = all(
            1 <= int(item.get("trl_level", 0)) <= 9 for item in assessments if isinstance(item, dict)
        )

        if basic_pass and fallback_consistency:
            quality = AssessmentQuality(
                coverage_complete=True,
                evidence_sufficient=True,
                consistency_ok=True,
                overall_pass=True,
                feedback=[],
            )
            existing_scores = state.get("quality_scores", {})
            if isinstance(existing_scores, dict):
                existing_scores["assessment_quality"] = quality.model_dump()
            else:
                existing_scores = {"assessment_quality": quality.model_dump()}
            return {
                "phase": "quality_passed",
                "quality_gate": "pass",
                "quality_scores": existing_scores,
            }

        # 기본 휴리스틱 실패 시 LLM Judge로 2차 판정
        judge = self.llm.with_structured_output(AssessmentQuality).invoke(
            [
                ("system", "너는 TRL 평가 품질 검토자다."),
                (
                    "human",
                    f"다음 TRL 평가 품질을 판정하라.\n\n"
                    f"- expected_count: {expected_count}\n"
                    f"- actual_count: {len(assessments)}\n"
                    f"- avg_evidence: {avg_evidence:.2f}\n"
                    f"- assessments: {json.dumps(assessments, ensure_ascii=False)[:12000]}",
                ),
            ]
        )

        if judge.overall_pass:
            existing_scores = state.get("quality_scores", {})
            if isinstance(existing_scores, dict):
                existing_scores["assessment_quality"] = judge.model_dump()
            else:
                existing_scores = {"assessment_quality": judge.model_dump()}
            return {
                "phase": "quality_passed",
                "quality_gate": "pass",
                "quality_scores": existing_scores,
            }

        # 재수집 루프 제한
        if state["iteration_count"] < state["max_iterations"]:
            existing_scores = state.get("quality_scores", {})
            if isinstance(existing_scores, dict):
                existing_scores["assessment_quality"] = judge.model_dump()
            else:
                existing_scores = {"assessment_quality": judge.model_dump()}
            return {
                "phase": "quality_failed",
                "quality_gate": "fail",
                "revision_feedback": judge.feedback,
                "iteration_count": state["iteration_count"] + 1,
                "quality_scores": existing_scores,
            }

        existing_scores = state.get("quality_scores", {})
        if isinstance(existing_scores, dict):
            existing_scores["assessment_quality"] = judge.model_dump()
        else:
            existing_scores = {"assessment_quality": judge.model_dump()}
        return {
            "phase": "quality_failed_max_iteration",
            "quality_gate": "pass",
            "errors": [*state.get("errors", []), "QualityCheck 재수집 최대 반복 초과"],
            "quality_scores": existing_scores,
        }


class ReportSynthesizerNode(BaseNode):
    """최종 보고서 생성 노드."""

    @node_exception_handler(node_name="report_synthesizer", fallback_goto="save_output")
    @log_execution_time("report_synthesizer")
    @ensure_state_keys(["trl_assessments", "threat_analysis", "collected_data", "references"])
    def run(self, state: dict[str, Any]) -> dict[str, Any]:
        feedback = state.get("revision_feedback", [])
        system_prompt = self.prompts.read("report_synthesizer_system.txt")
        citation_catalog = self._build_citation_catalog(state)
        retrieval_metrics = self._extract_retrieval_metrics(state)
        rag_sources = set(str(url).strip() for url in state.get("rag_sources", []) if str(url).strip())

        report_markdown = self.llm.invoke(
            [
                ("system", system_prompt),
                (
                    "human",
                    self._build_report_prompt(
                        collected_data=state["collected_data"],
                        assessments=state["trl_assessments"],
                        threats=state["threat_analysis"],
                        references=state.get("references", []),
                        citation_catalog=citation_catalog,
                        retrieval_metrics=retrieval_metrics,
                        feedback=feedback,
                    ),
                ),
            ]
        ).content

        report_text = str(report_markdown).strip()
        # LLM이 생성한 REFERENCE/REFERENCES 섹션(번호/형식 혼재 포함)은 모두 제거하고,
        # 마지막에 단일 `## REFERENCE` 섹션을 재구성한다.
        report_text = self._strip_existing_reference_sections(report_text)
        report_text = self._enforce_business_focus_sections(report_text)
        report_text = self._inject_retrieval_metrics(report_text, retrieval_metrics)
        report_text = self._inject_trl_rationale(
            report_text=report_text,
            assessments=state.get("trl_assessments", []),
            citation_catalog=citation_catalog,
        )
        if "[" not in report_text:
            report_text = report_text + "\n\n> 참고 근거 인용 번호가 누락되어 자동으로 보강되었습니다. [1]"

        report_text = self._replace_reference_section(report_text, citation_catalog, rag_sources=rag_sources)

        return {
            "phase": "report_drafted",
            "report_draft": report_text,
            "revision_feedback": [],
        }

    def _build_report_prompt(
        self,
        collected_data: dict[str, list[dict[str, Any]]],
        assessments: list[dict[str, Any]],
        threats: list[dict[str, Any]],
        references: list[str],
        citation_catalog: list[dict[str, Any]],
        retrieval_metrics: dict[str, Any],
        feedback: list[str],
    ) -> str:
        feedback_text = "\n".join(f"- {item}" for item in feedback) if feedback else "없음"
        return (
            f"[Collected Data]\n{json.dumps(collected_data, ensure_ascii=False, indent=2)[:12000]}\n\n"
            f"[TRL Assessments]\n{json.dumps(assessments, ensure_ascii=False, indent=2)}\n\n"
            f"[Threat Analysis]\n{json.dumps(threats, ensure_ascii=False, indent=2)}\n\n"
            f"[Citation Catalog]\n{json.dumps(citation_catalog, ensure_ascii=False, indent=2)}\n\n"
            f"[Retrieval Metrics]\n{json.dumps(retrieval_metrics, ensure_ascii=False, indent=2)}\n\n"
            f"[Revision Feedback]\n{feedback_text}\n\n"
            f"[References]\n{json.dumps(references[:40], ensure_ascii=False, indent=2)}\n\n"
            "=== 필수 구조 ===\n"
            "1. SUMMARY: 다음을 포함한 실행자 중심 요약\n"
            "   - 핵심 결론 3개 (각각 TRL 수준, 위협도, 경쟁사 비교 포함)\n"
            "   - 즉시 실행 액션 3개 (형식: 액션. (우선순위: X / 담당: Y / 기한: Z) [n])\n"
            "   - 모든 결론과 액션에 인용 번호 [n] 붙임\n"
            "2. 1. 분석 배경\n"
            "   - ### 목적성: HBM4/PIM/CXL이 전략적으로 중요한 이유를 명시\n"
            "3. 2. 분석 대상 기술 현황\n"
            "   - 각 기술별 TRL, 시장 규모, 기술 특성\n"
            "4. 3. 경쟁사 동향 분석\n"
            "   - ### 경쟁사별 비교: 다음 항목을 모두 포함한 표와 설명\n"
            "     * 기술 | 경쟁사 | TRL | 서브레벨 | 신뢰도 | 위협도 | 차별화 전략\n"
            "   - ### TRL 판단 근거: 각 (기술, 경쟁사) 조합별로 '왜 그 TRL인지'를 근거 2개 이상 + 정보 공백 1개 이상으로 제시\n"
            "5. 4. 전략적 시사점\n"
            "   - ### 종합 시사점: TRL 기반 순위, 위협 수준별 대응 계획, 투자 우선순위\n"
            "6. ### 신뢰성 메트릭\n"
            "   - Hit@1, Hit@3, Hit@5, MRR 값이 포함된 표\n"
            "7. REFERENCE: (작성하지 말 것) 본문에서 [n] 인용만 유지하라. REFERENCE 섹션은 시스템이 생성한다.\n\n"
            "=== 중요 요구사항 ===\n"
            "- 모든 주장에 [n] 인용 번호를 부착하라\n"
            "- 경쟁사별 기술 성숙도를 정량적으로 비교하는 표를 포함하라\n"
            "- TRL 4~6 수준 평가에는 '추정 기반' 또는 '간접 지표 기반'이라는 레이블을 명시하라\n"
            "- SUMMARY의 액션은 구체적이고 실행 가능해야 한다 (예: 'R&D 투자 확대'는 불충분, '200억원 추가 투자 승인, R&D팀, 2026년 2분기'는 충분)\n"
            "- 기술별 목적성: AI 메모리 병목 현상 해결, 데이터센터 성능 극대화, 에너지 효율 개선 등을 구체적으로 설명하라\n"
            "- 종합 시사점은 비교 분석 결과를 직접 인용하여 전략적 우선순위를 도출하라\n"
            "- REFERENCE는 모든 [n] 번호가 본문의 인용과 일치해야 하며, APA 형식을 엄격히 따르라"
        )

    def _extract_retrieval_metrics(self, state: dict[str, Any]) -> dict[str, Any]:
        scores = state.get("quality_scores", {}) if isinstance(state.get("quality_scores", {}), dict) else {}
        metrics = scores.get("retrieval_metrics", {})
        if not isinstance(metrics, dict):
            metrics = {}
        return {
            "retriever": metrics.get("retriever", "qdrant_similarity"),
            "num_queries": int(metrics.get("num_queries", 0) or 0),
            "Hit@1": float(metrics.get("Hit@1", 0.0) or 0.0),
            "Hit@3": float(metrics.get("Hit@3", 0.0) or 0.0),
            "Hit@5": float(metrics.get("Hit@5", 0.0) or 0.0),
            "MRR": float(metrics.get("MRR", 0.0) or 0.0),
        }

    def _enforce_business_focus_sections(self, report_text: str) -> str:
        """요구된 비즈니스 중심 소제목을 강제한다."""
        patched = report_text
        if "목적성" not in patched:
            patched = patched.replace(
                "## 1. 분석 배경",
                "## 1. 분석 배경\n### 목적성\n- 왜 지금 HBM4/PIM/CXL에 집중해야 하는지와 사업적 필요성을 명시한다.\n",
            )
        if "경쟁사별 비교" not in patched:
            patched = patched.replace(
                "## 3. 경쟁사 동향 분석",
                "## 3. 경쟁사 동향 분석\n### 경쟁사별 비교\n- 경쟁사별 TRL과 위협 수준을 표와 근거로 비교한다.\n",
            )
        if "종합 시사점" not in patched:
            patched = patched.replace(
                "## 4. 전략적 시사점",
                "## 4. 전략적 시사점\n### 종합 시사점\n- 비교 분석 결과를 사업 실행 계획으로 연결한다.\n",
            )
        return patched

    def _inject_retrieval_metrics(self, report_text: str, metrics: dict[str, Any]) -> str:
        """보고서에 Hit@1/3/5, MRR 메트릭 표를 삽입한다."""
        if all(token in report_text for token in ["Hit@1", "Hit@3", "Hit@5", "MRR"]):
            return report_text

        table = (
            "\n### 신뢰성 메트릭\n"
            "| Retriever | Queries | Hit@1 | Hit@3 | Hit@5 | MRR |\n"
            "|---|---:|---:|---:|---:|---:|\n"
            f"| {metrics.get('retriever','qdrant_similarity')} | {metrics.get('num_queries',0)} | "
            f"{metrics.get('Hit@1',0.0):.4f} | {metrics.get('Hit@3',0.0):.4f} | "
            f"{metrics.get('Hit@5',0.0):.4f} | {metrics.get('MRR',0.0):.4f} |\n"
        )

        return report_text.rstrip() + "\n" + table

    def _inject_trl_rationale(
        self,
        report_text: str,
        assessments: list[dict[str, Any]],
        citation_catalog: list[dict[str, Any]],
    ) -> str:
        """TRL 판단 근거 섹션을 보고서에 삽입한다(없으면)."""
        if "TRL 판단 근거" in report_text:
            return report_text
        block = self._build_trl_rationale_block(assessments=assessments, citation_catalog=citation_catalog)
        if not block:
            return report_text

        insert_pattern = re.compile(r"^[ \t]*#{1,6}\s*4\.\s*전략적 시사점", flags=re.MULTILINE)
        match = insert_pattern.search(report_text)
        if not match:
            return report_text.rstrip() + "\n\n" + block
        return (
            report_text[: match.start()].rstrip()
            + "\n\n"
            + block
            + "\n\n"
            + report_text[match.start() :].lstrip()
        )

    def _build_trl_rationale_block(
        self,
        assessments: list[dict[str, Any]],
        citation_catalog: list[dict[str, Any]],
    ) -> str:
        """TRL 판단 근거 섹션 마크다운을 생성한다."""
        if not assessments:
            return ""

        def as_float(value: Any, default: float = 0.0) -> float:
            try:
                return float(value)
            except Exception:  # noqa: BLE001
                return default

        lines: list[str] = []
        lines.append("### TRL 판단 근거")
        for item in assessments:
            if not isinstance(item, dict):
                continue
            technology = str(item.get("technology", "")).strip()
            competitor = str(item.get("competitor", "")).strip()
            trl_level = str(item.get("trl_level", "")).strip()
            trl_sublevel = str(item.get("trl_sublevel", "")).strip()
            basis = str(item.get("assessment_basis", "")).strip()
            confidence = as_float(item.get("confidence", 0.0), 0.0)
            note = str(item.get("assessment_note", "")).strip()
            evidence = item.get("evidence", [])
            gaps = item.get("information_gaps", [])

            header = f"- **{technology} / {competitor}**: TRL {trl_level} ({trl_sublevel}), basis={basis}, confidence={confidence:.2f}"
            lines.append(header)
            if note:
                lines.append(f"  - 판단 요약: {note}")

            if isinstance(evidence, list) and evidence:
                lines.append("  - 근거:")
                for ev in evidence[:3]:
                    ev_text = str(ev).strip()
                    if not ev_text:
                        continue
                    ev_text = self._append_catalog_citation(ev_text, citation_catalog)
                    lines.append(f"    - {ev_text}")

            if isinstance(gaps, list) and gaps:
                gap_texts = [str(g).strip() for g in gaps if str(g).strip()]
                if gap_texts:
                    lines.append(f"  - 정보 공백: {', '.join(gap_texts[:3])}")

        return "\n".join(lines).strip()

    def _append_catalog_citation(self, text: str, citation_catalog: list[dict[str, Any]]) -> str:
        """evidence 문장에 포함된 URL을 citation_catalog와 매칭해 [id] 인용을 덧붙인다."""
        if re.search(r"\[\d+\]", text):
            return text

        urls = re.findall(r"https?://\\S+", text)
        if not urls:
            return text

        def normalize(url: str) -> str:
            cleaned = url.strip()
            cleaned = cleaned.rstrip(").,;]>\"'")
            return cleaned.rstrip("/")

        normalized_urls = [normalize(url) for url in urls]
        for entry in citation_catalog:
            try:
                entry_id = int(entry.get("id", 0))
            except Exception:  # noqa: BLE001
                continue
            entry_url = normalize(str(entry.get("url", "")).strip())
            if not entry_url or entry_id <= 0:
                continue
            for candidate in normalized_urls:
                if not candidate:
                    continue
                if candidate == entry_url or candidate.startswith(entry_url) or entry_url.startswith(candidate):
                    return f"{text} [{entry_id}]"
        return text

    def _build_citation_catalog(self, state: dict[str, Any]) -> list[dict[str, Any]]:
        """보고서 본문에서 사용할 번호 기반 인용 카탈로그를 생성한다."""
        candidate_items = state.get("reference_items", [])
        if not candidate_items:
            candidate_items = [{"title": "Untitled", "url": url, "published_date": "", "company": "Unknown"} for url in state.get("references", [])]

        catalog: list[dict[str, Any]] = []
        seen_urls: set[str] = set()
        for item in candidate_items:
            url = str(item.get("url", "")).strip()
            if not url or url in seen_urls:
                continue
            seen_urls.add(url)
            catalog.append(
                {
                    "id": len(catalog) + 1,
                    "title": item.get("title", "Untitled"),
                    "url": url,
                    "published_date": item.get("published_date", ""),
                    "organization": item.get("company", "Unknown"),
                }
            )
            if len(catalog) >= 40:
                break
        return catalog

    def _strip_existing_reference_sections(self, report_text: str) -> str:
        """보고서에서 기존 REFERENCE/REFERENCES 섹션을 모두 제거한다."""
        text = report_text
        reference_heading = re.compile(
            r"^[ \t]*#{1,6}\s*(?:\d+\.\s*)?REFERENCE(?:S)?\s*$",
            flags=re.IGNORECASE | re.MULTILINE,
        )
        any_heading = re.compile(r"^[ \t]*#{1,6}\s+", flags=re.MULTILINE)

        while True:
            match = reference_heading.search(text)
            if not match:
                break
            start = match.start()
            next_heading = any_heading.search(text, pos=match.end())
            end = next_heading.start() if next_heading else len(text)
            text = (text[:start].rstrip() + "\n\n" + text[end:].lstrip()).strip()

        return text.strip()

    def _replace_reference_section(
        self,
        report_text: str,
        citation_catalog: list[dict[str, Any]],
        rag_sources: set[str] | None = None,
    ) -> str:
        """REFERENCE 섹션을 번호+APA 스타일로 정규화하고, WEB/RAG로 분리해 출력한다."""
        report_text = self._strip_existing_reference_sections(report_text)
        normalized_text, ordered_sources = self._normalize_citation_numbers(
            report_text, citation_catalog, rag_sources=rag_sources
        )
        apa_lines = self._build_apa_references(ordered_sources)
        marker = "## REFERENCE"
        return f"{normalized_text.rstrip()}\n\n{marker}\n{self._format_reference_sections(apa_lines, ordered_sources, rag_sources)}"

    def _normalize_citation_numbers(
        self,
        report_text: str,
        citation_catalog: list[dict[str, Any]],
        rag_sources: set[str] | None = None,
    ) -> tuple[str, list[dict[str, Any] | None]]:
        """본문 인용 번호를 [1..N]으로 재배열하고 대응 출처 목록을 만든다."""
        found_ids: list[int] = []
        for match in re.finditer(r"\[(\d+)\]", report_text):
            value = int(match.group(1))
            if value > 0 and value not in found_ids:
                found_ids.append(value)

        if not found_ids:
            return report_text, [citation_catalog[0]] if citation_catalog else [None]

        catalog_map = {int(item.get("id", 0)): item for item in citation_catalog if int(item.get("id", 0)) > 0}
        unused_catalog = [catalog_map[key] for key in sorted(catalog_map.keys())]
        old_to_new: dict[int, int] = {}

        source_by_old: dict[int, dict[str, Any] | None] = {}
        for old_id in found_ids:
            source = catalog_map.get(old_id)
            if source is None and unused_catalog:
                source = unused_catalog.pop(0)
            source_by_old[old_id] = source

        rag_sources = rag_sources or set()

        def is_rag(source: dict[str, Any] | None) -> bool:
            if not source:
                return False
            url = str(source.get("url", "")).strip()
            return bool(url and url in rag_sources)

        web_old_ids = [old_id for old_id in found_ids if not is_rag(source_by_old.get(old_id))]
        rag_old_ids = [old_id for old_id in found_ids if is_rag(source_by_old.get(old_id))]
        grouped_old_ids = [*web_old_ids, *rag_old_ids]

        chosen_sources: list[dict[str, Any] | None] = []
        for idx, old_id in enumerate(grouped_old_ids, start=1):
            old_to_new[old_id] = idx
            chosen_sources.append(source_by_old.get(old_id))

        def repl(match: re.Match[str]) -> str:
            old = int(match.group(1))
            new = old_to_new.get(old)
            return f"[{new}]" if new is not None else match.group(0)

        normalized_text = re.sub(r"\[(\d+)\]", repl, report_text)
        return normalized_text, chosen_sources

    def _format_reference_sections(
        self,
        apa_lines: list[str],
        ordered_sources: list[dict[str, Any] | None],
        rag_sources: set[str] | None,
    ) -> str:
        """REFERENCE를 WEB/RAG 두 섹션으로 나눠 출력한다."""
        rag_sources = rag_sources or set()
        web_lines: list[str] = []
        rag_lines: list[str] = []

        def is_rag_line(source: dict[str, Any] | None) -> bool:
            if not source:
                return False
            url = str(source.get("url", "")).strip()
            return bool(url and url in rag_sources)

        for line, source in zip(apa_lines, ordered_sources, strict=False):
            # Markdown 렌더러가 줄바꿈을 합쳐버리는 경우가 있어, blockquote + hard line break로 강제한다.
            formatted = f"> {line}  "
            if is_rag_line(source):
                rag_lines.append(formatted)
            else:
                web_lines.append(formatted)

        chunks: list[str] = []
        if web_lines:
            chunks.append("### WEB SEARCH")
            chunks.extend(web_lines)
        if rag_lines:
            if chunks:
                chunks.append("")
            chunks.append("### RAG (Vector DB)")
            chunks.extend(rag_lines)

        if not chunks:
            return "> [1] Author Unknown. (n.d.). Untitled. Retrieved from unknown source  "
        return "\n".join(chunks).rstrip()

    def _build_apa_references(self, ordered_sources: list[dict[str, Any] | None]) -> list[str]:
        """APA 스타일 레퍼런스를 생성한다 (저자/기관. 연도. 제목. 출처)."""
        if not ordered_sources:
            ordered_sources = [None]

        lines: list[str] = []
        for idx, item in enumerate(ordered_sources, start=1):
            if item is None:
                lines.append(f"[{idx}] Author Unknown. (n.d.). Untitled. Retrieved from unknown source")
                continue
            
            # 기관명/저자
            org = str(item.get("organization", "")).strip()
            if not org or org in ("Unknown", "N/A", ""):
                # URL 또는 제목에서 기관명 추출 시도
                url = str(item.get("url", "")).strip()
                if "arxiv" in url.lower():
                    org = "arXiv"
                elif "scholar.google" in url.lower():
                    org = "Google Scholar"
                elif "reuters" in url.lower():
                    org = "Reuters"
                elif "bloomberg" in url.lower():
                    org = "Bloomberg"
                elif "samsung" in url.lower():
                    org = "Samsung Semiconductor"
                elif "micron" in url.lower():
                    org = "Micron Technology"
                elif "skhynix" in url.lower():
                    org = "SK Hynix"
                else:
                    org = "Author Unknown"
            
            # 제목
            title = str(item.get("title", "")).strip()
            if not title:
                title = "Untitled research"
            
            # 연도
            date_raw = str(item.get("published_date", "")).strip()
            year = "n.d."
            if date_raw:
                try:
                    year = date_raw[:4] if len(date_raw) >= 4 and date_raw[:4].isdigit() else "n.d."
                except:  # noqa: E722
                    year = "n.d."
            
            # URL
            url = str(item.get("url", "")).strip() or ""
            url_part = f" Retrieved from {url}" if url else ""
            
            # APA 형식: 저자/기관. (연도). 제목. 출처.
            apa_line = f"[{idx}] {org}. ({year}). {title}."
            if url_part:
                apa_line += url_part
            lines.append(apa_line)
        
        return lines


class FinalReviewNode(BaseNode):
    """보고서 품질 검증 노드."""

    @node_exception_handler(node_name="final_review", fallback_goto="save_output")
    @log_execution_time("final_review")
    @ensure_state_keys(["report_draft", "iteration_count", "max_iterations"])
    def run(self, state: dict[str, Any]) -> dict[str, Any]:
        system_prompt = self.prompts.read("final_review_system.txt")
        evaluation = self.llm.with_structured_output(ReportEvaluation).invoke(
            [
                ("system", system_prompt),
                (
                    "human",
                    "보고서를 평가하라.\n\n"
                    f"{state['report_draft']}\n\n"
                    "체크 항목: 목적성, 경쟁사별 비교, 종합 시사점, 신뢰성 메트릭(Hit@1/3/5,MRR),"
                    " SUMMARY 실행가능성, 구조 완성도, 근거 인용,"
                    " 인용 번호-REFERENCE(APA) 일치, 논리 일관성, TRL 4~6 한계 인정.",
                ),
            ]
        )

        if evaluation.overall_pass:
            return {
                "phase": "review_passed",
                "quality_scores": {"final_review": evaluation.model_dump()},
                "review_decision": "approve",
            }

        if state["iteration_count"] < state["max_iterations"]:
            return {
                "phase": "review_failed",
                "quality_scores": {"final_review": evaluation.model_dump()},
                "revision_feedback": evaluation.feedback,
                "iteration_count": state["iteration_count"] + 1,
                "review_decision": "revise",
            }

        # 최대 반복을 초과하면 경고와 함께 저장 단계로 이동한다.
        return {
            "phase": "review_failed_max_iteration",
            "quality_scores": {"final_review": evaluation.model_dump()},
            "errors": [*state.get("errors", []), "최대 수정 반복 횟수 초과"],
            "review_decision": "warn",
        }


class EndWithWarningNode(BaseNode):
    """최대 반복 초과 등 경고 종료 노드."""

    @node_exception_handler(node_name="end_with_warning", fallback_goto="save_output")
    @log_execution_time("end_with_warning")
    def run(self, state: dict[str, Any]) -> dict[str, Any]:
        return {
            "phase": "ended_with_warning",
            "errors": [*state.get("errors", []), "end_with_warning"],
        }


class SaveOutputNode(BaseNode):
    """아티팩트 저장 노드."""

    def __init__(
        self,
        config: AppConfig,
        prompts: PromptRepository,
        llm: Any,
        logger: logging.Logger,
        output_service: OutputService,
    ) -> None:
        super().__init__(config=config, prompts=prompts, llm=llm, logger=logger)
        self.output_service = output_service

    @node_exception_handler(node_name="save_output", fallback_goto="__end__")
    @log_execution_time("save_output")
    def run(self, state: dict[str, Any]) -> dict[str, Any]:
        run_dir, report_path = self.output_service.save_state_artifacts(state)
        return {
            "phase": "completed",
            "run_dir": run_dir,
            "report_path": report_path,
        }

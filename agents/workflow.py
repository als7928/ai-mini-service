"""LangGraph 워크플로우 조립."""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from langchain_core.runnables.graph import MermaidDrawMethod
from langchain_core.runnables.graph_mermaid import draw_mermaid_png
from langgraph.graph import END, START, StateGraph

from .models import AppConfig, TechStrategyState, build_initial_state
from .nodes import (
    CompetitorProfilerNode,
    DomainKnowledgeNode,
    EndWithWarningNode,
    FinalReviewNode,
    PaperIngestorNode,
    QualityCheckNode,
    ReportSynthesizerNode,
    RequestParserNode,
    SaveOutputNode,
    TRLAssessorNode,
    TechnologyScannerNode,
)
from .prompts import PromptRepository
from .services import LLMFactory, OutputService, PaperIngestionService, TavilySearchService, VectorStoreService


class TechStrategyWorkflow:
    """전체 멀티 에이전트 그래프를 구성/실행한다."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

        self.prompts = PromptRepository(config.prompts_dir)
        self.llm = LLMFactory(config=config, logger=self.logger).create()
        self.search_service = TavilySearchService(config=config, logger=self.logger)
        self.ingestion_service = PaperIngestionService(logger=self.logger)
        self.vector_service = VectorStoreService(config=config, logger=self.logger)
        self.output_service = OutputService(config=config, logger=self.logger)

        self.request_parser_node = RequestParserNode(config, self.prompts, self.llm, self.logger)
        self.technology_scanner_node = TechnologyScannerNode(
            config, self.prompts, self.llm, self.logger, self.search_service
        )
        self.paper_ingestor_node = PaperIngestorNode(
            config,
            self.prompts,
            self.llm,
            self.logger,
            self.ingestion_service,
            self.vector_service,
        )
        self.domain_knowledge_node = DomainKnowledgeNode(
            config,
            self.prompts,
            self.llm,
            self.logger,
            self.vector_service,
        )
        self.competitor_profiler_node = CompetitorProfilerNode(config, self.prompts, self.llm, self.logger)
        self.trl_assessor_node = TRLAssessorNode(config, self.prompts, self.llm, self.logger)
        self.quality_check_node = QualityCheckNode(config, self.prompts, self.llm, self.logger)
        self.report_synthesizer_node = ReportSynthesizerNode(config, self.prompts, self.llm, self.logger)
        self.final_review_node = FinalReviewNode(config, self.prompts, self.llm, self.logger)
        self.end_with_warning_node = EndWithWarningNode(config, self.prompts, self.llm, self.logger)
        self.save_output_node = SaveOutputNode(
            config,
            self.prompts,
            self.llm,
            self.logger,
            self.output_service,
        )

        self.graph = self._build_graph()

    def _build_graph(self) -> Any:
        """StateGraph를 컴파일한다."""
        builder = StateGraph(TechStrategyState)
        builder.add_node("request_parser", self.request_parser_node.run)
        builder.add_node("technology_scanner", self.technology_scanner_node.run)
        builder.add_node("paper_ingestor", self.paper_ingestor_node.run)
        builder.add_node("domain_knowledge", self.domain_knowledge_node.run)
        builder.add_node("competitor_profiler", self.competitor_profiler_node.run)
        builder.add_node("trl_assessor", self.trl_assessor_node.run)
        builder.add_node("quality_check", self.quality_check_node.run)
        builder.add_node("report_synthesizer", self.report_synthesizer_node.run)
        builder.add_node("final_review", self.final_review_node.run)
        builder.add_node("end_with_warning", self.end_with_warning_node.run)
        builder.add_node("save_output", self.save_output_node.run)

        builder.add_edge(START, "request_parser")
        builder.add_edge("request_parser", "technology_scanner")
        builder.add_edge("technology_scanner", "paper_ingestor")
        builder.add_edge("paper_ingestor", "domain_knowledge")
        builder.add_edge("domain_knowledge", "competitor_profiler")
        builder.add_edge("competitor_profiler", "trl_assessor")
        builder.add_edge("trl_assessor", "quality_check")
        builder.add_conditional_edges(
            "quality_check",
            self._route_quality_check,
            {"pass": "report_synthesizer", "fail": "technology_scanner", "save_output": "save_output"},
        )
        builder.add_edge("report_synthesizer", "final_review")
        builder.add_conditional_edges(
            "final_review",
            self._route_final_review,
            {
                "approve": "save_output",
                "revise": "report_synthesizer",
                "warn": "end_with_warning",
                "save_output": "save_output",
            },
        )
        builder.add_edge("end_with_warning", "save_output")
        builder.add_edge("save_output", END)

        return builder.compile()

    def run(self, topic: str, user_request: str = "") -> dict[str, Any]:
        """워크플로우를 실행한다."""
        init_state = build_initial_state(
            topic=topic,
            user_request=user_request,
            max_iterations=self.config.max_iterations,
        )
        final_state = self.graph.invoke(init_state, {"recursion_limit": self.config.recursion_limit})
        return final_state

    def save_graph_image(self, output_path: str | Path | None = None) -> Path:
        """컴파일된 LangGraph 구조 이미지를 PNG로 저장한다."""
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = self.config.outputs_dir / f"langgraph_compiled_{timestamp}.png"
        else:
            path = Path(output_path)
            if not path.is_absolute():
                path = self.config.base_dir / path

        path.parent.mkdir(parents=True, exist_ok=True)
        graph_obj = self.graph.get_graph()
        mermaid_syntax = graph_obj.draw_mermaid()
        mermaid_syntax = mermaid_syntax.replace("graph TD;", "graph LR;")
        mermaid_syntax = mermaid_syntax.replace("curve: linear", "curve: basis")

        try:
            graph_image = draw_mermaid_png(
                mermaid_syntax=mermaid_syntax,
                max_retries=3,
                retry_delay=1.5,
            )
            path.write_bytes(graph_image)
            return path
        except Exception as api_error:  # noqa: BLE001
            self.logger.warning("Mermaid API 렌더링 실패, 로컬 렌더링 시도: %s", api_error)

        try:
            graph_image = draw_mermaid_png(
                mermaid_syntax=mermaid_syntax,
                draw_method=MermaidDrawMethod.PYPPETEER,
            )
            path.write_bytes(graph_image)
            return path
        except Exception as pyppeteer_error:  # noqa: BLE001
            self.logger.warning("Pyppeteer 렌더링 실패, mermaid 소스 저장: %s", pyppeteer_error)

        mermaid_path = path.with_suffix(".mmd")
        mermaid_path.write_text(mermaid_syntax, encoding="utf-8")
        return mermaid_path

    def _route_quality_check(self, state: TechStrategyState) -> str:
        """품질 게이트 분기 라우터."""
        if str(state.get("phase", "")) == "error":
            return "save_output"
        decision = str(state.get("quality_gate", "pass"))
        return decision if decision in {"pass", "fail"} else "pass"

    def _route_final_review(self, state: TechStrategyState) -> str:
        """최종 리뷰 분기 라우터."""
        if str(state.get("phase", "")) == "error":
            return "save_output"
        decision = str(state.get("review_decision", "approve"))
        return decision if decision in {"approve", "revise", "warn"} else "warn"

"""프롬프트 로더/렌더러."""

from __future__ import annotations

from pathlib import Path
from string import Template

from .exceptions import PromptTemplateError


DEFAULT_PROMPTS: dict[str, str] = {
    "request_parser_system.txt": (
        "너는 반도체 기술 전략 분석 요청을 구조화하는 파서다. "
        "요청에서 기술(HBM4/PIM/CXL)과 경쟁사(Samsung/Micron/SK hynix)를 식별하라."
    ),
    "scanner_system.txt": (
        "너는 기술 스캐너다. 최신 3개년 공개 정보를 우선 수집하고, "
        "출처 신뢰도를 고려해 요약을 작성하라."
    ),
    "domain_knowledge_system.txt": (
        "너는 도메인 지식 보강 에이전트다. TRL 정의와 기술 기본 개념을 간결히 보강하라."
    ),
    "competitor_profiler_system.txt": (
        "너는 경쟁사 프로파일러다. 공개 근거 기반으로만 프로파일을 만들고 "
        "추정은 명시적으로 표시하라."
    ),
    "trl_assessor_system.txt": (
        "너는 TRL 평가 에이전트다. TRL 4~6은 간접 지표 기반 추정임을 반드시 명시하라."
    ),
    "report_synthesizer_system.txt": (
        "너는 기술 전략 보고서 작성자다. SUMMARY/배경/기술현황/경쟁사동향/"
        "전략적 시사점/REFERENCE 형식을 반드시 지켜라. "
        "SUMMARY에는 즉시 실행 액션을 포함하고, 본문은 [n] 인용과 APA 레퍼런스를 사용하라. "
        "목적성/경쟁사별 비교/종합 시사점/신뢰성 메트릭(Hit@1/3/5, MRR)을 포함하라."
    ),
    "final_review_system.txt": (
        "너는 엄격한 품질 검토자다. 목적성/경쟁사 비교/종합 시사점/신뢰성 메트릭,"
        " SUMMARY 실행가능성, 구조 완성도, 근거, 인용-REFERENCE(APA) 정합성,"
        " 논리성, TRL 한계 고지 여부를 평가하라."
    ),
}


class PromptRepository:
    """`prompts/` 디렉토리에서 템플릿을 로딩한다."""

    def __init__(self, prompts_dir: Path) -> None:
        self.prompts_dir = prompts_dir
        self.prompts_dir.mkdir(parents=True, exist_ok=True)

    def read(self, filename: str) -> str:
        """프롬프트 파일 내용을 반환한다."""
        prompt_path = self.prompts_dir / filename
        if prompt_path.exists():
            return prompt_path.read_text(encoding="utf-8")
        if filename in DEFAULT_PROMPTS:
            return DEFAULT_PROMPTS[filename]
        raise PromptTemplateError(f"프롬프트 파일을 찾을 수 없습니다: {prompt_path}")

    def render(self, filename: str, **kwargs: str) -> str:
        """Template 문법($var)으로 렌더링한다."""
        template = Template(self.read(filename))
        try:
            return template.safe_substitute(**kwargs)
        except Exception as error:  # noqa: BLE001 - 프롬프트 렌더링 방어
            raise PromptTemplateError(f"프롬프트 렌더링 실패 ({filename}): {error}") from error

"""프로젝트 전용 예외 타입 모음."""


class TechStrategyError(Exception):
    """도메인 전용 기본 예외."""


class ConfigurationError(TechStrategyError):
    """환경 변수/설정 누락 예외."""


class SearchServiceError(TechStrategyError):
    """웹 검색 관련 예외."""


class IngestionError(TechStrategyError):
    """문서 수집/파싱 관련 예외."""


class VectorStoreError(TechStrategyError):
    """벡터 스토어 처리 예외."""


class PromptTemplateError(TechStrategyError):
    """프롬프트 템플릿 로딩/렌더링 예외."""


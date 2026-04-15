"""노드/서비스 공통 데코레이터."""

from __future__ import annotations

import logging
import time
from functools import wraps
from typing import Any, Callable, Iterable

def log_execution_time(name: str | None = None) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """함수 실행 시간을 로깅한다."""

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            start = time.perf_counter()
            try:
                return func(*args, **kwargs)
            finally:
                duration = time.perf_counter() - start
                logger = _resolve_logger(args)
                func_name = name or func.__name__
                logger.info("⏱️ %s 실행 시간: %.2fs", func_name, duration)

        return wrapper

    return decorator


def retry_on_exception(
    max_attempts: int = 3,
    delay_seconds: float = 1.0,
    exceptions: tuple[type[BaseException], ...] = (Exception,),
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """일시적 오류를 재시도한다."""

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            logger = _resolve_logger(args)
            last_error: BaseException | None = None

            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as error:  # noqa: PERF203 - 재시도 루프 의도
                    last_error = error
                    if attempt >= max_attempts:
                        break
                    logger.warning(
                        "재시도 %s/%s 실패 (%s): %s",
                        attempt,
                        max_attempts,
                        func.__name__,
                        error,
                    )
                    time.sleep(delay_seconds)

            assert last_error is not None
            raise last_error

        return wrapper

    return decorator


def ensure_state_keys(required_keys: Iterable[str]) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """노드 실행 전 필수 상태 키 존재 여부를 검증한다."""

    keys = tuple(required_keys)

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(func)
        def wrapper(self: Any, state: dict[str, Any], *args: Any, **kwargs: Any) -> Any:
            missing = [key for key in keys if key not in state]
            if missing:
                raise KeyError(f"필수 상태 키 누락: {', '.join(missing)}")
            return func(self, state, *args, **kwargs)

        return wrapper

    return decorator


def node_exception_handler(node_name: str, fallback_goto: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """노드 내부 예외를 캡처하고 안전하게 다음 노드로 핸드오프한다."""

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(func)
        def wrapper(self: Any, state: dict[str, Any], *args: Any, **kwargs: Any) -> Any:
            try:
                return func(self, state, *args, **kwargs)
            except Exception as error:  # noqa: BLE001 - 노드 안전성 우선
                logger = getattr(self, "logger", logging.getLogger(__name__))
                logger.exception("[%s] 노드 실행 실패: %s", node_name, error)

                current_errors = list(state.get("errors", []))
                current_errors.append(f"{node_name}: {error}")

                return {
                    "phase": "error",
                    "errors": current_errors,
                    "next_node": fallback_goto,
                    "report_draft": state.get("report_draft", "")
                    + f"\n\n> ⚠️ `{node_name}` 단계에서 오류가 발생했습니다: {error}",
                }

        return wrapper

    return decorator


def _resolve_logger(args: tuple[Any, ...]) -> logging.Logger:
    """self.logger가 있으면 사용하고, 없으면 모듈 기본 로거를 반환한다."""
    if args:
        candidate = args[0]
        logger = getattr(candidate, "logger", None)
        if isinstance(logger, logging.Logger):
            return logger
    return logging.getLogger(__name__)

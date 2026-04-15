"""Mini Project 실행 엔트리포인트."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from dotenv import load_dotenv

from agents import AppConfig, TechStrategyWorkflow


DEFAULT_TOPIC = (
    "HBM4, PIM, CXL 기술을 대상으로 Samsung/Micron/SK hynix 경쟁 동향과 "
    "TRL 기반 위협 수준을 분석해 전략 보고서를 작성하라."
)


def build_parser() -> argparse.ArgumentParser:
    """CLI 인자 파서."""
    parser = argparse.ArgumentParser(description="LangGraph 1.0 기반 기술 전략 분석 멀티 에이전트")
    parser.add_argument("--topic", type=str, default=DEFAULT_TOPIC, help="분석 토픽")
    parser.add_argument("--request", type=str, default="", help="추가 사용자 요청")
    parser.add_argument(
        "--save_graph",
        nargs="?",
        const="auto",
        default=None,
        help="컴파일된 LangGraph 이미지를 PNG로 저장 (경로 생략 시 outputs 자동 저장).",
    )
    parser.add_argument("--log-level", type=str, default="INFO", help="로그 레벨 (DEBUG/INFO/WARNING/ERROR)")
    return parser


def configure_logging(log_level: str) -> None:
    """애플리케이션 로깅 설정."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def main() -> None:
    """애플리케이션 실행."""
    parser = build_parser()
    args = parser.parse_args()

    configure_logging(args.log_level)
    logger = logging.getLogger("mini-project")

    try:
        base_dir = Path(__file__).resolve().parent
        load_dotenv(base_dir / ".env", override=False)

        config = AppConfig.from_env(base_dir)
        workflow = TechStrategyWorkflow(config=config)

        if args.save_graph is not None:
            graph_path = None if args.save_graph == "auto" else args.save_graph
            try:
                saved_path = workflow.save_graph_image(output_path=graph_path)
                logger.info("LangGraph 이미지 저장: %s", saved_path)
                print("graph_path:", saved_path)
            except Exception as graph_error:  # noqa: BLE001
                logger.warning("그래프 저장 실패(실행은 계속): %s", graph_error)

        final_state = workflow.run(topic=args.topic, user_request=args.request)

        logger.info("실행 완료")
        logger.info("Phase: %s", final_state.get("phase"))
        logger.info("Report: %s", final_state.get("report_path", ""))
        if final_state.get("errors"):
            logger.warning("오류 발생: %s", json.dumps(final_state["errors"], ensure_ascii=False))

        print("DONE")
        print("phase:", final_state.get("phase"))
        print("run_dir:", final_state.get("run_dir", ""))
        print("report_path:", final_state.get("report_path", ""))
    except Exception as error:  # noqa: BLE001 - CLI 단계에서 최종 보호
        logger.exception("실행 실패: %s", error)
        print("FAILED")
        print(str(error))


if __name__ == "__main__":
    main()

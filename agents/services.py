"""외부 연동/IO/벡터 처리 서비스 계층."""

from __future__ import annotations

import logging
import json
import os
import random
import re
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_openai import ChatOpenAI
from langchain_opendataloader_pdf import OpenDataLoaderPDFLoader
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings

try:
    from tavily import TavilyClient
except Exception:  # noqa: BLE001 - 선택 의존성(설치/환경에 따라 없을 수 있음)
    TavilyClient = None  # type: ignore[assignment]

from .decorators import log_execution_time, retry_on_exception
from .exceptions import ConfigurationError, IngestionError, VectorStoreError
from .models import AppConfig, IngestedPaper, SourceItem


class LLMFactory:
    """ChatOpenAI 생성 책임."""

    def __init__(self, config: AppConfig, logger: logging.Logger) -> None:
        self.config = config
        self.logger = logger

    @log_execution_time("llm_factory")
    def create(self) -> ChatOpenAI:
        """ChatOpenAI 모델 인스턴스를 생성한다."""
        if not os.getenv("OPENAI_API_KEY", "").strip():
            raise ConfigurationError("OPENAI_API_KEY가 비어 있습니다. `.env` 값을 확인하세요.")
        try:
            return ChatOpenAI(
                model=self.config.openai_model,
                temperature=self.config.openai_temperature,
            )
        except Exception as error:  # noqa: BLE001
            raise ConfigurationError(
                "ChatOpenAI 초기화에 실패했습니다. OPENAI_API_KEY 설정을 확인하세요."
            ) from error


class TavilySearchService:
    """Tavily 검색 래퍼."""

    DOMAIN_GROUPS: dict[str, list[str]] = {
        "academic": ["arxiv.org", "scholar.google.com"],
        "news": [
            "semianalysis.com",
            "eetimes.com",
            "tomshardware.com",
            "reuters.com",
            "bloomberg.com",
        ],
        "official": ["semiconductor.samsung.com", "micron.com", "skhynix.com"],
    }

    def __init__(self, config: AppConfig, logger: logging.Logger) -> None:
        self.config = config
        self.logger = logger
        self.enabled = True
        self.disabled_reason: str = ""
        if TavilyClient is None:
            self._disable("tavily-python 패키지를 찾을 수 없어 웹 검색을 비활성화합니다.")
            return
        api_key = os.getenv("TAVILY_API_KEY", "").strip()
        if not api_key:
            self._disable("TAVILY_API_KEY가 없어 웹 검색을 비활성화합니다.")
            return
        try:
            self.client = TavilyClient(api_key=api_key)
        except Exception as error:  # noqa: BLE001
            self._disable(f"Tavily 초기화 실패: {error}")

    @log_execution_time("technology_scanner_search")
    def search_by_technology(self, technology: str, competitors: list[str], topic: str) -> list[SourceItem]:
        """기술/경쟁사 기준으로 다층 검색을 수행한다."""
        if not self.enabled or self.client is None:
            return []

        results: list[SourceItem] = []
        query_candidates = self._build_queries(technology=technology, competitors=competitors, topic=topic)

        for query in query_candidates:
            for group_name, domains in self.DOMAIN_GROUPS.items():
                # Tavily 호출이 실패하더라도 전체 워크플로우는 계속 진행되어야 한다.
                raw_results = self._invoke_search(query=query, include_domains=domains)
                for item in self._normalize_results(raw_results):
                    source = SourceItem(
                        technology=technology,
                        company=self._extract_company(item.get("title", "") + " " + item.get("content", "")),
                        source_group=group_name,
                        query=query,
                        title=str(item.get("title", "Untitled")),
                        url=str(item.get("url", "")),
                        published_date=str(item.get("published_date", "")),
                        summary=str(item.get("content", ""))[:1000],
                    )
                    if source.url and source.summary:
                        results.append(source)

        deduped = self._deduplicate(results)
        return deduped

    def _invoke_search(self, query: str, include_domains: list[str]) -> list[dict[str, Any]]:
        """Tavily 단일 호출."""
        if not self.enabled or self.client is None:
            return []
        try:
            raw = self.client.search(
                query=query,
                include_domains=include_domains,
                max_results=self.config.tavily_max_results,
                search_depth="advanced",
                days=self.config.tavily_lookback_days,
                topic="news",
                include_answer=False,
                include_raw_content=False,
                include_images=False,
            )
            return self._normalize_results(raw)
        except Exception as error:  # noqa: BLE001
            # 사용량 제한/인증 오류 등은 재시도로 해결되지 않는 경우가 많으므로,
            # 즉시 Tavily를 비활성화하고 빈 결과로 폴백한다.
            self._disable(f"Tavily 검색 실패: {error}")
            return []

    def _disable(self, reason: str) -> None:
        """Tavily 검색 기능을 비활성화한다."""
        self.enabled = False
        self.client = None
        self.disabled_reason = reason
        self.logger.warning("%s", reason)

    def _build_queries(self, technology: str, competitors: list[str], topic: str) -> list[str]:
        """1차/2차 검색 질의를 구성한다."""
        base_topic = "semiconductor memory technology trend"
        basic = [f"{technology} {base_topic}", f"{technology} AI data center memory"]
        if topic:
            basic.append(f"{technology} {topic}")
        for competitor in competitors:
            basic.append(f"{competitor} {technology} roadmap")
            basic.append(f"{competitor} {technology} yield power latency")
        return basic

    def _normalize_results(self, raw: Any) -> list[dict[str, Any]]:
        """Tavily 응답을 리스트 형태로 정규화한다."""
        if isinstance(raw, list):
            return [item for item in raw if isinstance(item, dict)]
        if isinstance(raw, dict):
            inner = raw.get("results")
            if isinstance(inner, list):
                return [item for item in inner if isinstance(item, dict)]
        return []

    def _deduplicate(self, items: list[SourceItem]) -> list[SourceItem]:
        """URL 기준 중복 제거."""
        deduped: list[SourceItem] = []
        seen: set[str] = set()
        for item in items:
            if item.url in seen:
                continue
            seen.add(item.url)
            deduped.append(item)
        return deduped

    def _extract_company(self, text: str) -> str:
        text_lower = text.lower()
        if "samsung" in text_lower:
            return "Samsung"
        if "micron" in text_lower:
            return "Micron"
        if "sk hynix" in text_lower or "hynix" in text_lower:
            return "SK hynix"
        return "Unknown"

    def extract_arxiv_urls(self, items: list[SourceItem]) -> list[str]:
        """검색 결과에서 arXiv URL만 추출한다."""
        urls: list[str] = []
        pattern = re.compile(r"^(?:\d{4}\.\d{4,5}|[a-z\-]+\/\d{7})(?:v\d+)?$")
        for item in items:
            url = (item.url or "").strip()
            if "arxiv.org" not in url:
                continue

            parsed = urlparse(url)
            parts = [part for part in parsed.path.split("/") if part]
            if len(parts) < 2:
                continue

            kind, identifier = parts[0], parts[1]
            identifier = identifier.replace(".pdf", "")
            if kind not in {"abs", "pdf"}:
                continue
            if not pattern.match(identifier):
                continue

            normalized = f"https://arxiv.org/abs/{identifier}"
            urls.append(normalized)

        return sorted(set(urls))[:8]


class PaperIngestionService:
    """arXiv 다운로드/파싱 서비스."""

    def __init__(self, logger: logging.Logger) -> None:
        self.logger = logger

    @log_execution_time("paper_ingestion")
    def ingest_arxiv_urls(self, arxiv_urls: list[str], data_dir: Path) -> tuple[list[IngestedPaper], list[Document]]:
        """arXiv URL 목록을 PDF로 내려받아 파싱하고 문서 청크를 반환한다."""
        data_dir.mkdir(parents=True, exist_ok=True)
        ingested: list[IngestedPaper] = []
        docs: list[Document] = []

        for url in sorted(set(arxiv_urls)):
            pdf_url = self._build_pdf_url(url)
            if not pdf_url:
                ingested.append(
                    IngestedPaper(
                        url=url,
                        pdf_url="",
                        title=url,
                        status="failed",
                        error_message="arXiv URL 형식 파싱 실패",
                    )
                )
                continue

            try:
                pdf_path = self._download_pdf(pdf_url, data_dir=data_dir)
                paper_docs = self._load_pdf_documents(pdf_path)
                title = paper_docs[0].metadata.get("title", url) if paper_docs else url
                for doc in paper_docs:
                    metadata = dict(getattr(doc, "metadata", {}) or {})
                    metadata["source"] = url
                    metadata["pdf_url"] = pdf_url
                    metadata["file_path"] = str(pdf_path)
                    doc.metadata = metadata
                ingested.append(
                    IngestedPaper(
                        url=url,
                        pdf_url=pdf_url,
                        title=str(title),
                        num_chunks=len(paper_docs),
                        status="success",
                    )
                )
                docs.extend(paper_docs)
            except Exception as error:  # noqa: BLE001
                ingested.append(
                    IngestedPaper(
                        url=url,
                        pdf_url=pdf_url,
                        title=url,
                        status="failed",
                        error_message=str(error),
                    )
                )
                self.logger.warning("논문 수집 실패 (%s): %s", url, error)

        return ingested, docs

    def _build_pdf_url(self, source_url: str) -> str:
        """arXiv abs URL을 pdf URL로 변환한다."""
        if "arxiv.org" not in source_url:
            return ""
        if "/pdf/" in source_url and source_url.endswith(".pdf"):
            return source_url

        if "/abs/" in source_url:
            paper_id = source_url.split("/abs/")[-1].strip().rstrip("/")
        else:
            parsed = urlparse(source_url)
            paper_id = parsed.path.strip("/").split("/")[-1]

        paper_id = paper_id.replace(".pdf", "")
        if not paper_id:
            return ""
        return f"https://arxiv.org/pdf/{paper_id}.pdf"

    @retry_on_exception(max_attempts=2, delay_seconds=1.2, exceptions=(httpx.HTTPError,))
    def _download_pdf(self, pdf_url: str, data_dir: Path) -> Path:
        """PDF 파일 다운로드."""
        safe_name = re.sub(r"[^a-zA-Z0-9._-]+", "_", pdf_url).strip("_")
        output_path = data_dir / f"{safe_name[-100:]}.pdf"
        with httpx.Client(timeout=40.0, follow_redirects=True) as client:
            response = client.get(pdf_url)
            response.raise_for_status()
            output_path.write_bytes(response.content)
        return output_path

    def _load_pdf_documents(self, pdf_path: Path) -> list[Document]:
        """OpenDataLoader 우선, 실패 시 PyPDF로 폴백한다."""
        if not pdf_path.exists():
            raise IngestionError(f"PDF 파일이 존재하지 않습니다: {pdf_path}")

        try:
            loader = OpenDataLoaderPDFLoader(file_path=str(pdf_path), format="text", hybrid_fallback=True)
            docs = loader.load()
            if docs:
                return docs
        except Exception as first_error:  # noqa: BLE001
            self.logger.info("OpenDataLoader 파싱 실패, PyPDF로 폴백: %s", first_error)

        fallback_loader = PyPDFLoader(str(pdf_path))
        docs = fallback_loader.load()
        if not docs:
            raise IngestionError(f"PDF 파싱 결과가 비어 있습니다: {pdf_path}")
        return docs


class VectorStoreService:
    """Qdrant 벡터 스토어 서비스."""

    def __init__(self, config: AppConfig, logger: logging.Logger) -> None:
        self.config = config
        self.logger = logger
        self.embedding: Embeddings | None = None
        self.vectorstore: QdrantVectorStore | None = None
        self.indexed_chunks: list[Document] = []
        self.qdrant_path = config.data_dir / "qdrant_local"
        self.qdrant_path.mkdir(parents=True, exist_ok=True)

    def load_existing(self) -> bool:
        """기존 로컬 Qdrant 컬렉션을 로드한다(있으면)."""
        if self.vectorstore is not None:
            return True

        try:
            has_any_files = self.qdrant_path.exists() and any(self.qdrant_path.iterdir())
        except Exception:  # noqa: BLE001
            has_any_files = False
        if not has_any_files:
            return False

        if self.embedding is None:
            self.embedding = self._create_embedding_model()

        errors: list[str] = []

        # 1) qdrant-client를 직접 써서 로드 시도
        try:
            from qdrant_client import QdrantClient  # noqa: PLC0415

            client = QdrantClient(path=str(self.qdrant_path))
            try:
                self.vectorstore = QdrantVectorStore(
                    client=client,
                    collection_name=self.config.qdrant_collection_name,
                    embedding=self.embedding,
                )
            except TypeError:
                self.vectorstore = QdrantVectorStore(  # type: ignore[call-arg]
                    client=client,
                    collection_name=self.config.qdrant_collection_name,
                    embeddings=self.embedding,
                )
            self.logger.info("기존 Qdrant 로컬 컬렉션 로드: %s", self.config.qdrant_collection_name)
            return True
        except Exception as error:  # noqa: BLE001
            errors.append(str(error))

        # 2) langchain-qdrant 헬퍼가 있으면 로드 시도
        try:
            loader = getattr(QdrantVectorStore, "from_existing_collection", None)
            if loader is None:
                return False
            try:
                self.vectorstore = loader(
                    embedding=self.embedding,
                    collection_name=self.config.qdrant_collection_name,
                    path=str(self.qdrant_path),
                )
            except TypeError:
                self.vectorstore = loader(  # type: ignore[misc]
                    embeddings=self.embedding,
                    collection_name=self.config.qdrant_collection_name,
                    path=str(self.qdrant_path),
                )
            self.logger.info("기존 Qdrant 로컬 컬렉션 로드: %s", self.config.qdrant_collection_name)
            return True
        except Exception as error:  # noqa: BLE001
            errors.append(str(error))

        if errors:
            self.logger.warning("기존 Qdrant 로컬 로드 실패: %s", " / ".join(errors[:2]))
        return False

    def _create_embedding_model(self) -> Embeddings:
        """설정된 provider(huggingface|jina|voyage)에 따라 임베딩 모델을 생성한다."""
        provider = (self.config.embedding_provider or "huggingface").lower().strip()
        self.logger.info("임베딩 프로바이더: %s / 모델: %s", provider, self.config.embedding_model)

        if provider == "jina":
            return self._create_jina_embedding()
        if provider == "voyage":
            return self._create_voyage_embedding()
        return self._create_huggingface_embedding()

    def _create_huggingface_embedding(self) -> HuggingFaceEmbeddings:
        """HuggingFace 임베딩 (BGE-M3 우선, 실패 시 경량 모델로 폴백)."""
        try:
            return HuggingFaceEmbeddings(
                model_name=self.config.embedding_model,
                encode_kwargs={"normalize_embeddings": True},
            )
        except Exception as error:  # noqa: BLE001
            self.logger.warning("임베딩 모델 로드 실패(%s), 폴백 모델 사용: %s", self.config.embedding_model, error)
            return HuggingFaceEmbeddings(
                model_name=self.config.fallback_embedding_model,
                encode_kwargs={"normalize_embeddings": True},
            )

    def _create_jina_embedding(self) -> Embeddings:
        """Jina Embeddings API 기반 임베딩 모델을 생성한다."""
        api_key = self.config.jina_api_key or os.getenv("JINA_API_KEY", "")
        if not api_key:
            raise ConfigurationError(
                "JINA_API_KEY가 설정되지 않았습니다. .env에 JINA_API_KEY를 추가하세요."
            )
        try:
            from langchain_community.embeddings import JinaEmbeddings  # noqa: PLC0415

            # EMBEDDING_MODEL이 기본값(BGE-M3)이면 Jina 전용 기본 모델로 치환
            model = self.config.embedding_model
            if model == "BAAI/bge-m3":
                model = "jina-embeddings-v3"
            return JinaEmbeddings(jina_api_key=api_key, model_name=model)
        except ImportError as error:
            raise ConfigurationError(
                "langchain-community 패키지가 필요합니다: pip install langchain-community"
            ) from error

    def _create_voyage_embedding(self) -> Embeddings:
        """Voyage AI voyage-3-large 임베딩 모델을 생성한다."""
        api_key = self.config.voyage_api_key or os.getenv("VOYAGE_API_KEY", "")
        if not api_key:
            raise ConfigurationError(
                "VOYAGE_API_KEY가 설정되지 않았습니다. .env에 VOYAGE_API_KEY를 추가하세요."
            )
        try:
            from langchain_voyageai import VoyageAIEmbeddings  # noqa: PLC0415

            # EMBEDDING_MODEL이 기본값이면 Voyage 전용 기본 모델로 치환
            model = self.config.embedding_model
            if model == "BAAI/bge-m3":
                model = "voyage-3-large"
            return VoyageAIEmbeddings(voyage_api_key=api_key, model=model)
        except ImportError as error:
            raise ConfigurationError(
                "langchain-voyageai 패키지가 필요합니다: pip install langchain-voyageai"
            ) from error

    @log_execution_time("vectorstore_upsert")
    def build(self, documents: list[Document]) -> int:
        """문서를 청킹하고 Qdrant에 적재한다."""
        if not documents:
            raise VectorStoreError("벡터화할 문서가 없습니다.")

        if self.embedding is None:
            self.embedding = self._create_embedding_model()

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_documents(documents)
        if not chunks:
            raise VectorStoreError("청킹 결과가 비어 있습니다.")
        self.indexed_chunks = chunks

        try:
            self.vectorstore = QdrantVectorStore.from_documents(
                documents=chunks,
                embedding=self.embedding,
                path=str(self.qdrant_path),
                collection_name=self.config.qdrant_collection_name,
                force_recreate=True,
            )
            return len(chunks)
        except Exception as error:  # noqa: BLE001
            raise VectorStoreError(f"Qdrant 적재 실패: {error}") from error

    def retrieve(self, query: str, k: int) -> list[Document]:
        """유사도 검색."""
        if self.vectorstore is None:
            self.load_existing()
        if self.vectorstore is None:
            return []
        try:
            return self.vectorstore.similarity_search(query, k=k)
        except Exception as error:  # noqa: BLE001
            self.logger.warning("벡터 검색 실패: %s", error)
            return []

    def compute_retrieval_metrics(self, sample_size: int = 20) -> dict[str, Any]:
        """Retriever 성능 메트릭(Hit@1/3/5, MRR)을 계산한다."""
        if self.vectorstore is None or not self.indexed_chunks:
            return {
                "retriever": "qdrant_similarity",
                "num_queries": 0,
                "Hit@1": 0.0,
                "Hit@3": 0.0,
                "Hit@5": 0.0,
                "MRR": 0.0,
            }

        sampled = random.sample(self.indexed_chunks, k=min(sample_size, len(self.indexed_chunks)))
        hit1 = hit3 = hit5 = 0
        reciprocal_ranks: list[float] = []

        for chunk in sampled:
            query = self._build_eval_query(chunk)
            retrieved = self.retrieve(query=query, k=5)
            rank = self._find_relevant_rank(chunk, retrieved)
            if rank is not None:
                if rank <= 1:
                    hit1 += 1
                if rank <= 3:
                    hit3 += 1
                if rank <= 5:
                    hit5 += 1
                reciprocal_ranks.append(1.0 / rank)
            else:
                reciprocal_ranks.append(0.0)

        n = max(1, len(sampled))
        return {
            "retriever": "qdrant_similarity",
            "num_queries": len(sampled),
            "Hit@1": round(hit1 / n, 4),
            "Hit@3": round(hit3 / n, 4),
            "Hit@5": round(hit5 / n, 4),
            "MRR": round(sum(reciprocal_ranks) / n, 4),
        }

    def _build_eval_query(self, chunk: Document) -> str:
        """청크 기반 평가용 질의를 생성한다."""
        content = (chunk.page_content or "").replace("\n", " ").strip()
        words = [token for token in content.split(" ") if token][:28]
        prefix = " ".join(words)
        title = str(chunk.metadata.get("title", "")).strip()
        technology = str(chunk.metadata.get("technology", "")).strip()
        query = " ".join(part for part in [title, technology, prefix] if part).strip()
        return query[:300] or "HBM4 PIM CXL technology comparison"

    def _find_relevant_rank(self, target: Document, retrieved: list[Document]) -> int | None:
        """검색 결과에서 정답 청크의 최초 순위를 찾는다."""
        target_text = (target.page_content or "").strip()
        target_source = str(target.metadata.get("source", "")).strip()

        for idx, doc in enumerate(retrieved, start=1):
            doc_text = (doc.page_content or "").strip()
            doc_source = str(doc.metadata.get("source", "")).strip()
            if target_text and doc_text == target_text:
                return idx
            if target_source and doc_source and target_source == doc_source:
                overlap = self._token_overlap_ratio(target_text, doc_text)
                if overlap >= 0.6:
                    return idx
        return None

    def _token_overlap_ratio(self, text_a: str, text_b: str) -> float:
        """두 텍스트의 토큰 중첩 비율."""
        tokens_a = set(token.lower() for token in text_a.split() if token)
        tokens_b = set(token.lower() for token in text_b.split() if token)
        if not tokens_a or not tokens_b:
            return 0.0
        inter = len(tokens_a & tokens_b)
        base = max(1, len(tokens_a))
        return inter / base


class OutputService:
    """결과물 저장 서비스."""

    def __init__(self, config: AppConfig, logger: logging.Logger) -> None:
        self.config = config
        self.logger = logger
        self.config.outputs_dir.mkdir(parents=True, exist_ok=True)

    def create_run_dir(self) -> Path:
        """타임스탬프 실행 디렉토리를 생성한다."""
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = self.config.outputs_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir

    @log_execution_time("save_outputs")
    def save_state_artifacts(self, state: dict[str, Any]) -> tuple[str, str]:
        """보고서/상태를 파일로 저장한다."""
        run_dir = self.create_run_dir()
        report_path = run_dir / "report.md"
        state_path = run_dir / "state.json"
        errors_path = run_dir / "errors.log"
        latest_report_path = self.config.outputs_dir / "report.md"

        report_text = state.get("report_draft", "").strip()
        if not report_text:
            report_text = "# 보고서 생성 실패\n\n오류 목록을 확인하세요."
        report_path.write_text(report_text, encoding="utf-8")
        latest_report_path.write_text(report_text, encoding="utf-8")

        serialized_state = json.dumps(state, ensure_ascii=False, indent=2, default=str)
        state_path.write_text(serialized_state, encoding="utf-8")

        if state.get("errors"):
            errors_path.write_text("\n".join(state["errors"]), encoding="utf-8")

        return str(run_dir), str(report_path)


def validate_required_env() -> None:
    """필수 키 검증."""
    if not str(Path.cwd()):
        raise ConfigurationError("작업 디렉토리를 확인할 수 없습니다.")
    if not (Path.cwd() / ".env").exists():
        # .env가 없는 환경도 허용하지만, 경고를 위한 예외 대신 pass 처리
        return

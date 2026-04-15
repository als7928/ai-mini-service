## Subject

글로벌 AI 반도체 시장의 급격한 성장과 함께 HBM,
PIM, CXL을 중심으로 한 차세대 메모리 기술 경쟁이 심화되고 있다. 따라서 반도체 기업은 HBM4 이후 세대까지 지속하기 위해 경쟁사의 연구 방향과 기술 성숙도를 실시간으로 파악해야 할 필요가 있다.
본 프로젝트는 이러한 요구에 부응하여, LangGraph 기반 Multi-Agent Agentic Workflow를 활용한 “기술 전략 분석 보고서” 자동 생성 솔루션을 제공한다. 동작 과정은 다음같다. 오픈소스 LLM과 Tavily Search API를 활용하여 arXiv, Google Scholar, 뉴스 사이트로부터 최근 3개년의 학술 논문 및 기술 정보를 자동 수집하고, 수집된 논문 PDF를 파싱하여 벡터 스토어에 색인한 뒤 RAG 기반 분석을 수행하는 에이전트들을 둔다. 각 에이전트는 상호 협력하며, 경쟁사별 기술 성숙도와 위협 수준을 정량적으로 분석하고, 담당자가 즉시 활용 가능한 전략 보고서를 산출한다. 


## Overview

- **Objective** : HBM4 / PIM / CXL 차세대 반도체 기술에 대한 경쟁사(Samsung, Micron)의 최신 R&D 동향을 자동 수집·분석하여, SK hynix R&D 담당자가 즉시 활용 가능한 기술 전략 분석 보고서를 생성하는 Multi-Agent Agentic Workflow 시스템 구현
- **Method** : LangGraph 기반 Distributed Multi-Agent 패턴 — 웹 검색 → PDF 파싱 → 벡터 색인(RAG) → 경쟁사 프로파일링 → TRL 정량 평가 → 보고서 합성의 파이프라인을 에이전트 간 직접 핸드오프로 구성
- **Tools** : Tavily Web Search (웹·학술 정보 수집), arXiv PDF Ingestor (논문 다운로드·파싱), Qdrant Similarity Search (RAG 벡터 검색)

---

## Features
- Tavily Search API를 통한 arXiv · Google Scholar · 뉴스 사이트에서의 실시간 정보 수집
- PDF 논문 자동 다운로드 및 파싱 (1컬럼 / 2컬럼 레이아웃, 표·수식 구분)
- Qdrant 벡터 스토어 기반 RAG — 정적 배경 지식과 동적 최신 논문을 통합 색인
- Corrective RAG 패턴 적용 (관련성 임계값 미달 시 쿼리 재작성 → 재검색 → 웹 폴백)
- TRL(Technology Readiness Level 1~9) 프레임워크 기반 기술 성숙도 정량 평가
- 경쟁사별 위협 수준 분석 및 전략적 시사점 도출
- 본문 인용 + APA 스타일 REFERENCE 자동 생성
- 품질 검증 및 최종 리뷰 자기 교정 루프

---

## Tech Stack

| Category | Details |
|---|---|
| Language | Python 3.11+ |
| Agent Framework | LangGraph ≥ 1.0, LangChain ≥ 1.0 |
| LLM | ChatOpenAI (GPT-4o / gpt-4o-mini) |
| Web Search | Tavily Search API (`langchain-tavily`) |
| PDF Parsing | `langchain-opendataloader-pdf` ≥ 2.0 |
| Embedding | BAAI/bge-m3 (`sentence-transformers`) |
| Vector Store | Qdrant (`langchain-qdrant`) |
| Data Validation | Pydantic (structured output) |
| Package Manager | uv |

---

## Agents

| Agent | Role |
|---|---|
| **RequestParser** | 사용자 요청을 파싱하여 분석 대상 기술·경쟁사·분석 초점을 추출 |
| **TechnologyScanner** | arXiv·Google Scholar·뉴스에서 최신 기술 정보 수집 |
| **PaperIngestor** | 수집된 논문 PDF를 다운로드·파싱하여 Qdrant 벡터 스토어에 동적 색인 |
| **DomainKnowledge** | RAG로 TRL 프레임워크·HBM/PIM/CXL 배경 지식 색인 & 컨텍스트 보강 |
| **CompetitorProfiler** | 수집 정보를 기반으로 경쟁사별 기술 프로파일 구성 |
| **TRLAssessor** | 간접 지표를 근거로 각 기술의 성숙도(TRL 1~9)와 위협 수준을 정량 평가 |
| **QualityCheck** | 수집 품질 체크 — 기준 미달 시 TechnologyScanner로 재수집 루프 |
| **ReportSynthesizer** | 분석 결과를 종합하여 SUMMARY·경쟁사 동향·전략적 시사점 포함 보고서 초안 생성 |
| **FinalReview** | 보고서 품질 최종 검토 후 승인(approve) / 수정(revise) / 경고(warn) 분기 |
| **SaveOutput** | 최종 보고서 및 실행 상태를 `outputs/` 디렉터리에 저장 |

---


## Directory Structure
```text
mini-project/
├── data/        # PDF 문서 / Qdrant 로컬 데이터
├── agents/      # Agent 모듈 (OOP + LangGraph 노드)
├── prompts/     # 프롬프트 템플릿
├── outputs/     # 실행 결과 보고서/상태 저장
├── app.py       # 실행 스크립트
└── README.md
```



## 환경 변수 (`.env`)
- `OPENAI_API_KEY`: ChatOpenAI 호출에 필요
- `TAVILY_API_KEY`: Technology Scanner 웹 검색에 필요 (없어도 실행은 가능)
- 선택:
  - `OPENAI_MODEL`
  - `OPENAI_TEMPERATURE`
  - `MAX_ITERATIONS`
  - `EMBEDDING_MODEL`
  - `FALLBACK_EMBEDDING_MODEL`

## 실행 방법
```bash
source .venv/bin/activate
python app.py --topic "HBM4/PIM/CXL 경쟁사 분석"
```

```bash
# 컴파일된 LangGraph 이미지 저장 (사용자 지정 경로)
python app.py --save_graph outputs/my_graph.png
```


## Contributors

- **안민혁** : 시스템 아키텍처 설계, LangGraph 워크플로우 구현, 에이전트 노드 개발, 데코레이터 공통 처리 구현
- **고길훈** : RAG 파이프라인 구현, 벡터 스토어 구성, 보고서 합성 에이전트 개발, 프롬프트 템플릿 설계

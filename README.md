# RAG Agent System

## 프로젝트 개요

한국 법률 4지선다 객관식 문제(KMMLU Criminal-Law & Law) 풀이를 위한 Production 수준의 RAG(Retrieval-Augmented Generation) 시스템이다.  
`gpt-4o-mini` + `text-embedding-3-small` 기반으로, 무분별한 검색을 지양하고 유용한 데이터만 제공하도록 유도하며, LLM의 Hallucination을 최소화하는 프롬프트 및 컨텍스트 경량화에 집중하였다.

> 데이터셋 출처: https://huggingface.co/datasets/HAERAE-HUB/KMMLU

## Agent System 구조와 실행 방법

Tech Stack: `FastAPI`, `Docker`  
Vector DB: `ChromaDB`

### 프로젝트 구조
```text
.
├── Dockerfile
├── Makefile
├── README.md
├── accuracy.png
├── config.py
├── data
│   ├── chroma_db
│   │   └── chroma.sqlite3
│   ├── collections.py
│   ├── dev.csv
│   ├── loader.py
│   ├── test.csv
│   └── train.csv
├── eval
│   ├── eval_dev.py
│   └── eval_test.py
├── logs
│   ├── eval-dev.log
│   └── eval.log
├── main.py
├── pyproject.toml
├── pyrightconfig.json
├── ruff.toml
├── service
│   ├── agent
│   │   ├── dto
│   │   │   ├── agent_request.py
│   │   │   └── agent_response.py
│   │   ├── openai_client.py
│   │   ├── openai_service.py
│   │   └── prompt
│   │       ├── response_schema.py
│   │       └── system_prompt.py
│   └── retrieval
│       ├── dto
│       │   └── retrieval_dto.py
│       └── retrieval_service.py
└── uv.lock
```

### 실행 방법

**1. 필수 패키지 설치**  
본인 로컬 환경에 `docker`, `uv`, `make`가 실행되도록 설치해주세요

**2. 가상 환경 실행**

다음 명령어로 가상 환경을 실행합니다. (Mac OS 기준)

```bash
source .venv/bin/activate
```

**3. 환경 변수 설정**  

.env 파일을 프로젝트 루트에 다음과 같이 생성하고, 값을 채워 넣어주세요
```text
# ENV should follow these strings ['DEBUG', 'INFO', 'WARN', 'CRITICAL']
ENV=

# OpenAI API token
OPENAI_API_KEY=
```

**4. 데이터 세팅**  

아래 명령을 입력하여 필요한 데이터셋을 자동으로 ChromaDB에 임베딩하여 적재합니다.
단, 반드시 `data`디렉토리에 `train.csv`가 존재해야 합니다.

```bash
make dataloader
```

**5. 빌드**  
도커 프로세스가 실행중인지 확인하고, 아래 명령을 입력하여 도커 이미지를 빌드해주세요.
```bash
make build
```

**6. 실행**  

아래 명령을 입력하여 로컬 환경에 RAG Agent System을 실행해주세요.

```bash
make run
```

**6. 평가**  

다른 콘솔 프로세스를 띄워서 평가 스크립트를 실행해야 합니다.
해당 어플리케이션은 `data/train.csv`와 `data/dev.csv`를 기준으로 개발되었습니다. 
때문에 `test.csv`를 준비하되, 위 두가지 포맷과 동일하게 제작하여 `data` 디렉토리 내부에 배치해주세요
이후, 프로젝트 루트에서 다음 명령을 실행하여 Docker container에서 서빙되는 API를 호출하는 평가 스크립트가 실행됩니다.

```bash
make evaluate-test
```

**7. 환경 제거**

평가가 종료되었으면 컨테이너 및 이미지를 다음 명령어를 입력하여 제거합니다.

```bash
make clean
```

### 개발자 가이드

**1. 환경 변수 설정**  
.env 파일을 프로젝트 루트에 다음과 같이 생성해주세요
```text
# ENV should follow these strings ['DEBUG', 'INFO', 'WARN', 'CRITICAL']
ENV=

# OpenAI API token
OPENAI_API_KEY=
```

**2. 데이터 세팅**  

아래 명령을 입력하여 필요한 데이터셋을 자동으로 ChromaDB에 임베딩하여 적재합니다.
단, 반드시 `data`디렉토리에 `train.csv`가 존재해야 합니다.

```bash
make dataloader
```

**3. 평가 실행**  
개발 후 RAG 시스템을 평가하기 위해 아래 명령을 실행해주세요
단, 반드시 `data`디렉토리에 `dev.csv`가 존재해야 합니다.

```bash
make evaluate-dev
```

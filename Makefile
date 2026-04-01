SHELL = bash
ifneq ("$(wildcard .env)","")
	include .env
endif

.PHONY: lint format type-check dataloader pr evaluate-dev evaluate-test build run clean

.ONESHELL:
lint:
	uv run ruff check --fix --exit-non-zero-on-fix --config=ruff.toml

.ONESHELL:
format:
	uv run ruff format --check --config=ruff.toml

.ONESHELL:
type-check:
	uv run pyright

.ONESHELL:
pr: lint format type-check

.ONESHELL:
dataloader:
	uv run python -m data.loader

.ONESHELL:
evaluate-dev:
	uv run python -m eval.eval_dev

.ONESHELL:
evaluate-test:
	uv run python -m eval.eval_test data/test.csv

.ONESHELL:
build:
	docker build -t rag-agent-system .

.ONESHELL:
run:
	docker run \
		--rm --name rag-agent \
		--env-file .env \
		-v ./logs:/app/logs \
		-v ./data:/app/data \
		-p 8000:8000 \
		rag-agent-system

.ONESHELL:
clean: 
	docker stop rag-agent
	@while [ -n "$$(docker ps -aq -f name=^/rag-agent$$)" ]; do \
		echo "Waiting for container to be completely removed..."; \
		sleep 1; \
	done
	docker image rm rag-agent-system

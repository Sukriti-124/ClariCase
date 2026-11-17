# All targets run from the platform/ directory. Compose uses this dir as its
# project root; build contexts are the repo root (one level up).

SHELL := /bin/bash
DC    := docker compose
PY    := python
REPO_ROOT := $(shell cd .. && pwd)

.PHONY: help build up up-ensemble up-agentic down logs ps test smoke smoke-ensemble clean

help:
	@echo "Targets:"
	@echo "  build   Build all service images"
	@echo "  up      Start the stack (detached)"
	@echo "  up-ensemble  Start the stack plus embed-api and ensemble-api, routed through the gate"
	@echo "  up-agentic   As up-ensemble, plus agentic-api (LLM stage) wired in"
	@echo "  down    Stop and remove containers (keeps pgdata volume)"
	@echo "  logs    Tail logs from all services"
	@echo "  ps      List running services"
	@echo "  test    Run per-service unit tests on the host"
	@echo "  smoke   End-to-end smoke test (brings compose up + curls /predict + tears down)"
	@echo "  smoke-ensemble  Same, through ensemble-api (profile ensemble)"
	@echo "  clean   Down + remove pgdata volume"

build:
	$(DC) build

up:
	@[[ -f .env ]] || cp .env.example .env
	$(DC) up -d

# Needs docs/metrics/improved_model_2/model.joblib,
# docs/metrics/ensemble_stack/stack.joblib and data/cache/knn_embed/index.npz
# (see README, "Ensemble profile").
up-ensemble:
	@[[ -f .env ]] || cp .env.example .env
	PRIMARY_URL=http://ensemble-api:8000 $(DC) --profile ensemble up -d --build

# Needs AGENTIC_API_KEY in .env and data/cache/agentic/example_pool.*
up-agentic:
	@[[ -f .env ]] || cp .env.example .env
	PRIMARY_URL=http://ensemble-api:8000 AGENTIC_URL=http://agentic-api:8000 PRIMARY_TIMEOUT_S=12 \
	  $(DC) --profile ensemble --profile agentic up -d --build

down:
	$(DC) --profile ensemble --profile agentic down

logs:
	$(DC) logs -f

ps:
	$(DC) ps

test:
	cd $(REPO_ROOT) && $(PY) -m pytest platform/shared/tests -v
	cd $(REPO_ROOT) && $(PY) -m pytest platform/services/svm-api/tests -v
	cd $(REPO_ROOT) && $(PY) -m pytest platform/services/router/tests -v
	cd $(REPO_ROOT) && $(PY) -m pytest platform/services/embed-api/tests -v
	cd $(REPO_ROOT) && $(PY) -m pytest platform/services/ensemble-api/tests -v
	cd $(REPO_ROOT) && $(PY) -m pytest platform/services/agentic-api/tests -v

smoke:
	bash tests/smoke.sh

smoke-ensemble:
	bash tests/smoke_ensemble.sh

clean:
	$(DC) --profile ensemble --profile agentic down -v

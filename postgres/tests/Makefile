# tests/Makefile.in
POSTGRES_SOURCE = @POSTGRES_SOURCE@
POSTGRES_INSTALL = @POSTGRES_INSTALL@
POSTGRES_DATA = @POSTGRES_DATA@
POSTGRES_USER = @POSTGRES_USER@

# Specify disabled tests here (space-separated list)
DISABLED_TESTS = pubmed_table numeric_test basic_parallel_scan bluesky_queries

.PHONY: test clean $(patsubst sql/%.sql,test-%,$(wildcard sql/*.sql)) list-tests pytest pytest-parallel pytest-coverage

# Run SQL-based tests (original bash/SQL infrastructure)
test:
	@./scripts/run_tests.sh --disabled-tests "$(DISABLED_TESTS)"

# Run Python-based tests
pytest:
	@echo "Running Python tests with pytest..."
	@cd py_tests && export PYTHONPATH=. && pytest -v

# Run Python tests in parallel
pytest-parallel:
	@echo "Running Python tests in parallel..."
	@cd py_tests && export PYTHONPATH=. && pytest -v -n auto

# Run Python tests with coverage
pytest-coverage:
	@echo "Running Python tests with coverage..."
	@cd py_tests && export PYTHONPATH=. && pytest -v --cov=. --cov-report=html --cov-report=term
	@echo "Coverage report generated in py_tests/htmlcov/index.html"

# Generate test targets for each SQL file
$(patsubst sql/%.sql,test-%,$(wildcard sql/*.sql)): test-%: sql/%.sql
	@./scripts/run_tests.sh --disabled-tests "$(DISABLED_TESTS)" $*

clean:
	@rm -rf logs/*
	@rm -rf results/*
	@rm -rf py_tests/htmlcov/
	@rm -rf py_tests/.pytest_cache/
	@rm -rf py_tests/.coverage
	@find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	@echo "Cleaned test logs, results, and Python artifacts"

# Show available tests
list-tests:
	@echo "SQL Tests (bash/SQL infrastructure):"
	@echo "===================================="
	@for test in sql/*.sql; do \
		testname=$${test#sql/}; \
		testname=$${testname%.sql}; \
		skip=false; \
		for disabled in $(DISABLED_TESTS); do \
			if [ "$$testname" = "$$disabled" ]; then \
				skip=true; \
				break; \
			fi; \
		done; \
		if [ "$$skip" = "true" ]; then \
			echo "  $$testname (DISABLED)"; \
		else \
			echo "  $$testname"; \
		fi \
	done
	@echo ""
	@echo "Python Tests (pytest infrastructure):"
	@echo "======================================"
	@cd py_tests && pytest --collect-only -q 2>/dev/null || echo "  (run 'pip install -r py_tests/requirements.txt' first)"
	@echo ""
	@echo "Usage:"
	@echo "  make test              - Run SQL tests"
	@echo "  make pytest            - Run Python tests"
	@echo "  make pytest-parallel   - Run Python tests in parallel"
	@echo "  make pytest-coverage   - Run Python tests with coverage"

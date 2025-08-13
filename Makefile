# Makefile (for use with GNU Make)
# Define which folders to include
INCLUDE_DIRS = encodec/trainer encodec/baseline_data

# Exclude specific files or folders (e.g., 'trainers/old/' and 'trainers/experimental/')
TARGETS = $(shell find $(INCLUDE_DIRS) -name "*.py")

.PHONY: format lint

format:
	poetry run black $(TARGETS)
	poetry run isort $(TARGETS)
	poetry run ruff check $(TARGETS) --fix

lint:
	poetry run ruff $(TARGETS)
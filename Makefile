# Makefile (for use with GNU Make)
# Define which folders to include
INCLUDE_DIRS = ppg trainers

# Exclude specific files or folders (e.g., 'trainers/old/' and 'trainers/experimental/')
TARGETS = $(shell find $(INCLUDE_DIRS) -name "*.py" \
	| grep -v "trainers/mtsm/MOMENT/" \
	| grep -v "trainers/mtsm/TS2VEC/" \
	| grep -v "trainers/downstream/test_downstream_classification.py" \
	| grep -v "trainers/downstream/classification.py")

.PHONY: format lint

format:
	poetry run black $(TARGETS)
	poetry run isort $(TARGETS)
	poetry run ruff check $(TARGETS) --fix

lint:
	poetry run ruff $(TARGETS)
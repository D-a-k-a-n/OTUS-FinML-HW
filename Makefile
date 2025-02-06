#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = otus_finml_hw
PYTHON_VERSION = 3.10.15
PYTHON_INTERPRETER = python

#################################################################################
# COMMANDS                                                                      #
#################################################################################


## Install Python Dependencies
.PHONY: requirements
requirements:
	$(PYTHON_INTERPRETER) -m pip install -U pip uv
	$(PYTHON_INTERPRETER) -m uv pip install -r requirements.txt
	



## Delete all compiled Python files
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

## Format source code with black
.PHONY: format
format:
	python -m black --config pyproject.toml src

## Set up Python interpreter environment
.PHONY: create_environment
create_environment:
	@bash -c "if pyenv virtualenvs | grep -q '$(PROJECT_NAME)'; then \
		echo '>>> Environment $(PROJECT_NAME) already exists. Activating it...'; \
	else \
		pyenv virtualenv $(PYTHON_VERSION) $(PROJECT_NAME); \
		echo '>>> New pyenv virtualenv $(PROJECT_NAME) created.'; \
	fi"
	@echo ">>> Activate the environment with: pyenv activate $(PROJECT_NAME)"




#################################################################################
# PROJECT RULES                                                                 #
#################################################################################


## Make Dataset
.PHONY: data
data: requirements
	$(PYTHON_INTERPRETER) src/dataset.py $(ARGS)


## Make Plots App
.PHONY: plots
plots: requirements
	$(PYTHON_INTERPRETER) src/plots.py $(ARGS)

## Analyze data for quality and anomalies
.PHONY: analyze
analyze: requirements
	$(PYTHON_INTERPRETER) src/analyze.py $(ARGS)


#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys; \
lines = '\n'.join([line for line in sys.stdin]); \
matches = re.findall(r'\n## (.*)\n[\s\S]+?\n([a-zA-Z_-]+):', lines); \
print('Available rules:\n'); \
print('\n'.join(['{:25}{}'.format(*reversed(match)) for match in matches]))
endef
export PRINT_HELP_PYSCRIPT

help:
	@$(PYTHON_INTERPRETER) -c "${PRINT_HELP_PYSCRIPT}" < $(MAKEFILE_LIST)

#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = OTUS-HW1
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

## Lint using flake8 and black (use `make format` to do formatting)
.PHONY: lint
lint:
	flake8 otus_hw1
	isort --check --diff --profile black otus_hw1
	black --check --config pyproject.toml otus_hw1

## Format source code with black
.PHONY: format
format:
	black --config pyproject.toml otus_hw1




## Set up python interpreter environment
.PHONY: create_environment
create_environment:
	@bash -c "if pyenv versions | grep -q $(PROJECT_NAME); then \
		echo '>>> Environment $(PROJECT_NAME) already exists. Activating it...'; \
	else \
		pyenv virtualenv $(PYTHON_INTERPRETER) $(PROJECT_NAME); \
		echo '>>> New pyenv virtualenv $(PROJECT_NAME) created.'; \
	fi"
	@echo ">>> Activate the environment with:\npyenv activate $(PROJECT_NAME)"

	



#################################################################################
# PROJECT RULES                                                                 #
#################################################################################


## Make Dataset
.PHONY: data
data: requirements
	pwd
	$(PYTHON_INTERPRETER) otus_hw1/dataset.py


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

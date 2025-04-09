# How to begin writting .sh 
#! /bin/bash

# Run tests
ruff --check .
mypy .
black . --check 
bandit -r . -f txt



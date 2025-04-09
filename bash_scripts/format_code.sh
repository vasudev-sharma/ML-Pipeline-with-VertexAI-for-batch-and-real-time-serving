# How to begin writting .sh 
#! /bin/bash

# Run tests

printf "Running Ruff........\n\n"
ruff check . --exit-zero

printf "\n\nRunning MyPy.....\n\n"
mypy .

printf "\n\nRunning black.....\n\n"
black . --check 

prinf "\n\nRunning bandit.....\n\n"
bandit -r . -f txt --exit-zero



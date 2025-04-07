# How to begin writting .sh 

# Run tests
flake8 .
pylint **/*.py --exit-zero
bandit -r . -f txt
mypy . 

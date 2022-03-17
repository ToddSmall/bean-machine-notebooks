# Experimenting with [Bean Machine](https://beanmachine.org)

See the notebooks in the `notebooks` directory.

## Installation

- `/usr/local/bin/python3 -m venv --upgrade-deps .`
- `source bin/activate`
- `python -m pip install pip-tools`
- `pip-compile requirements.in`
- `pip-compile dev-requirements.in`
- `pip-sync requirements.txt dev-requirements.txt`

## Altair Figures

Alas, the Altair figures in the notebooks will not render on GitHub. However, the figures to appear correctly when the notebooks are rendered with [nbviewer](https://nbviewer.org).
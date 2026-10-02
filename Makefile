# Atalhos do projeto (requer uv: https://docs.astral.sh/uv/)
.PHONY: setup data train notebooks run-notebooks app test lint clean

setup:            ## cria o ambiente e instala dependências (inclui extras dev e contextual)
	uv sync --extra dev --extra contextual

data:             ## baixa e prepara dados públicos
	uv run replang download && uv run replang prepare

train:            ## treina modelos locais sobre o corpus Machado
	uv run replang train all

train-fast:
	uv run replang train all --fast

notebooks:        ## reconstrói os .ipynb a partir de notebooks/_src
	uv run replang notebooks build

run-notebooks:    ## executa todos os notebooks (grava saídas)
	uv run replang notebooks both

app:              ## interface Streamlit
	uv run replang app

test:
	uv run pytest -q

lint:
	uv run ruff check src tests scripts app && uv run ruff format --check src tests scripts app

clean:
	rm -rf data/cache/* .pytest_cache

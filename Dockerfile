FROM python:3.11-slim

RUN mkdir /Senlac
RUN mkdir /Senlac/results
COPY data src pyproject.toml /Senlac/
WORKDIR /Senlac
RUN pip install .
CMD [ "run_crm", "/Senlac/results/results.csv"]

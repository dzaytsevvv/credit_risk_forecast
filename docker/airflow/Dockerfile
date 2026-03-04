FROM apache/airflow:2.9.3-python3.11

USER root
RUN apt-get -o Acquire::Retries=5 update \
  && apt-get -o Acquire::Retries=5 install -y --no-install-recommends libgomp1 \
  && rm -rf /var/lib/apt/lists/*

USER airflow
RUN pip install --no-cache-dir --retries 5 --timeout 60 \
  pandas \
  numpy \
  scikit-learn \
  lightgbm \
  mlflow \
  pyyaml \
  pyarrow \
  matplotlib \
  joblib \
  psycopg2-binary

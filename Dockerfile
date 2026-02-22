FROM python:3.11-slim

WORKDIR /opt/project

COPY requirements.txt /opt/project/requirements.txt
RUN pip install --no-cache-dir -r /opt/project/requirements.txt

COPY src /opt/project/src
COPY scripts /opt/project/scripts

ENV PYTHONPATH=/opt/project/src

RUN chmod +x /opt/project/scripts/entrypoint.sh

CMD ["bash", "/opt/project/scripts/entrypoint.sh"]

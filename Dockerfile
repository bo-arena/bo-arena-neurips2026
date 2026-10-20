# Build for linux/amd64 and push to a registry:
#   docker buildx build --platform linux/amd64 -t <user>/bo-arena:<tag> --push .
# then convert on the cluster with:
#   singularity pull images/bo_arena.sif docker://<user>/bo-arena@sha256:<digest>
FROM python:3.14.8@sha256:12d187a59ecdeb95df0a78cd6107a84398455365a66cc2183db2be42b6067d33

COPY pyproject.toml uv.lock /opt/bo-arena/
WORKDIR /opt/bo-arena

RUN pip install --root-user-action=ignore uv==0.13.0 && \
    UV_PROJECT_ENVIRONMENT=/opt/venv uv sync --locked --no-dev --group experiments --no-install-project

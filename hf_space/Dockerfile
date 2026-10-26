# The Space runs as a Docker Space (README front matter: sdk: docker, app_port: 7860).
# Why not the Gradio SDK: Hugging Face's Gradio launcher starts the UI on port 7860 itself,
# so app.py's combined FastAPI (+ mounted Gradio) server could not bind the port
# ("[Errno 98] address already in use") and only the old Gradio UI was served.
FROM python:3.13-slim

# git: requirements.txt installs the word checker from GitHub (git+https)
RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    HF_HOME=/home/user/.cache/huggingface PORT=7860
WORKDIR /home/user/app

# CPU-only torch first (the default PyPI wheel pulls several GB of CUDA libraries)
RUN pip install --user --index-url https://download.pytorch.org/whl/cpu "torch>=2.1"
COPY --chown=user requirements.txt .
RUN pip install --user -r requirements.txt
COPY --chown=user . .

EXPOSE 7860
CMD ["python", "app.py"]

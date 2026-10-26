# Optional: only used if this Space is switched to `sdk: docker` (README
# front matter, plus `app_port: 7860`). Under the current `sdk: gradio`,
# Spaces runs `python app.py`, which starts the same FastAPI+Gradio server.
FROM python:3.13-slim
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH PYTHONUNBUFFERED=1
WORKDIR /home/user/app
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt
COPY --chown=user . .
EXPOSE 7860
CMD ["python", "app.py"]

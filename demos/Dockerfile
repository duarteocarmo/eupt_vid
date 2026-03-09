FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*

COPY index.html fasttext.js fasttext_wasm.js fasttext_wasm.wasm ./

RUN curl -L -o model_quantized.ftz \
    https://huggingface.co/duarteocarmo/fasttext-euptvid/resolve/main/model_quantized.ftz

EXPOSE 8080

CMD ["python", "-m", "http.server", "8080"]

FROM huggingface/transformers-pytorch-gpu:4.35.0
RUN CMAKE_ARGS="-DLLAMA_CUBLAS=on" FORCE_CMAKE=1 pip install llama-cpp-python==0.2.11
COPY requirements.txt .
RUN pip install -r requirements.txt
WORKDIR /app
CMD ["python3", "main.py"]
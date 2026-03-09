# Base image with PyTorch and CUDA
FROM pytorch/pytorch:2.10.0-cuda13.0-cudnn9-runtime
WORKDIR /app
# Install dependencies (torch is already provided by the base image)
COPY requirements.txt .
ENV PIP_BREAK_SYSTEM_PACKAGES=1
RUN grep -v '^torch' requirements.txt > /tmp/requirements.txt && \
    pip install -r /tmp/requirements.txt
COPY . .
# Optional: install a small English spaCy model if NER is needed
# RUN python -m spacy download en_core_web_sm
CMD ["python", "api.py"]
EXPOSE 9004


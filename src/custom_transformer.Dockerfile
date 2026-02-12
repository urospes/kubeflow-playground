FROM python:3.12-slim

RUN pip install --no-cache-dir kserve==0.15.2 kserve-storage cloudpickle scikit-learn pandas numpy

COPY utils/inference_preprocessor.py /app/inference_preprocessor.py

ENTRYPOINT ["python", "/app/inference_preprocessor.py"]
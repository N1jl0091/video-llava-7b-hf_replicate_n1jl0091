FROM python:3.10-slim

WORKDIR /src

# Install system dependencies
RUN apt-get update && apt-get install -y ffmpeg

# Install Python package to handle Hugging Face caching
RUN pip3 install huggingface_hub

# Copy the uncompressed weights directly
COPY weights /src/weights

# Copy your project files
COPY . /app

WORKDIR /app

RUN pip install --no-cache-dir -r requirements.txt

# We don't need to install Python or copy requirements here
# Cog will handle Python package installation

# We'll let Cog handle the Python package installation
# COPY requirements.txt .
# RUN pip install --no-cache-dir -r requirements.txt

# We don't need to copy the code here, Cog will do that

# Remove the CMD instruction, Cog will handle running the predictor
# CMD ["python", "predict.py"]

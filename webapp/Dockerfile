# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies needed for OpenCV and Torch on ARM/Slim
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0t64 \
    libgl1 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and related tools
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Copy the requirements file into the container
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

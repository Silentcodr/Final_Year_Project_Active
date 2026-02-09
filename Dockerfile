
# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Install system dependencies for OpenCV and Tesseract
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Create upload directory if it doesn't exist (crucial for file operations)
RUN mkdir -p static/upload

# Expose port 5000 to the world outside this container
EXPOSE 5000

# Define environment variable
ENV FLASK_APP=main.py

# Run main.py using gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "main:app"]

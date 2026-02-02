# Use official Python image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy project files
COPY . /app

# Install pip dependencies (add requirements.txt if available)
# RUN pip install -r requirements.txt

# Install required packages directly (edit as needed)
RUN pip install langchain-core langchain-ollama langchain-chroma langchain-community langchain-text-splitters chromadb

# Expose port if running a web server (optional)
# EXPOSE 8000

# Default command (edit as needed)
CMD ["python", "RR.py"]

# Use official Python image
FROM python:3.11-slim

# Create a non-root user for safety
RUN useradd -m appuser

# Set work directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.eval.txt ./
RUN pip install --no-cache-dir -r requirements.eval.txt

# Switch to non-root user
USER appuser

# Default command (can be overridden)
CMD ["python", "test.py"]

# Use official Python slim image — smaller and faster
FROM python:3.11-slim

# Set working directory inside container
WORKDIR /app

# Copy requirements first — Docker caches this layer
# so it only reinstalls packages when requirements change
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app.py .
COPY train.py .

# Copy trained model artefacts
COPY model/ ./model/

# Expose the port Flask runs on
EXPOSE 5001

# Run with gunicorn for production
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:5001", "--workers", "2"]
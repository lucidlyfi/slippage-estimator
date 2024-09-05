# Use an official Python runtime as a parent image
FROM python:3.10-alpine

# Install Poetry
RUN pip install --upgrade pip && pip install --no-cache-dir poetry 

# Set the working directory in the container
WORKDIR /app

# Copy the pyproject.toml and poetry.lock files
COPY pyproject.toml poetry.lock /app/

# Install dependencies
RUN poetry install --no-root 

# Copy the rest of the application code
COPY . /app

# Expose port 5000 to the outside world
EXPOSE 5000

# Run the Flask application
CMD ["poetry", "run", "gunicorn", "--workers", "3", "--bind", "0.0.0.0:5000", "app:app"]

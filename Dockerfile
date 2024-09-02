# Use an Alpine PyPy base image
FROM pypy:3-slim

# Install dependencies
RUN pip install --upgrade pip && pip install --no-cache-dir poetry 

# Set the working directory
WORKDIR /app

# Copy only the pyproject.toml and poetry.lock files first
COPY pyproject.toml poetry.lock /app/

# Install Python dependencies
RUN poetry install 

# Copy the rest of the application code
COPY . /app

# Expose port 5000 to the outside world
EXPOSE 5000

# Run the Flask application
CMD ["poetry", "run", "flask", "run", "--host=0.0.0.0", "--port=5000"]


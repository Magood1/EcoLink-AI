# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Install poetry
RUN pip install poetry

# Copy the dependency files to the working directory
COPY pyproject.toml poetry.lock ./

# Install project dependencies
RUN poetry config virtualenvs.create false && poetry install --no-root --no-dev

# Copy the rest of the application code to the working directory
COPY . .

# Command to run the application (e.g., the smoke test)
CMD ["python", "train.py", "--config", "configs/smoke_test.yaml", "--num-episodes", "1"]


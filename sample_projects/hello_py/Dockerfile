FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Copy requirements.txt and install dependencies if it exists
# If your project has dependencies, create a requirements.txt file
# e.g., echo "some-package" > requirements.txt
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip pip install --no-cache-dir -r requirements.txt || echo "No requirements.txt found or dependencies already installed."

# Copy the application code into the container
COPY hello.py .

# Command to run the hello.py script
CMD ["python", "hello.py"]
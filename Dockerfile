### THIS DOCKERFILE IS PURELY FOR TESTING PURPOSES ###
### TO SEE IF THE PROGRAM RUNS OUTSIDE THE WINDOWS OS ###

# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file
COPY requirements.txt .

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Run the train.py script when the container launches
CMD ["python", "train.py"]
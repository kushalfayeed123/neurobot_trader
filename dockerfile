# 1. Use a lightweight Python image
FROM python:3.10-slim

# 2. Set the working directory inside the container
WORKDIR /app

# 3. Copy only the requirements first (to speed up builds)
COPY requirements.txt .

# 4. Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy the rest of your code (engine.py, trainer.py, .env, etc.)
COPY . .

# 6. Start the engine by default
CMD ["python", "engine.py"]
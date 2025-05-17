#!/bin/bash

# Step 1: Pull latest changes
echo "📥 Pulling latest changes..."
git pull

# Step 2: Run pytest and show + save output
echo "🧪 Running pytest..."
python3 -m pytest -v | tee /media/sf_AI-stock-agent-main/pytest.txt

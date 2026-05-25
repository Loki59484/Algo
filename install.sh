#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

echo "=================================================="
echo "🚀 INITIATING TRADING ENGINE SETUP..."
echo "=================================================="

# 1. Update System Packages
echo "📦 Updating Ubuntu system packages..."
sudo apt update && sudo apt upgrade -y

# 2. Create Required Directories
echo "📁 Creating data and logs directories..."
mkdir -p ~/Algo/data
mkdir -p ~/Algo/logs

# 3. Install Miniconda (if not already installed)
if [ ! -d "$HOME/miniconda3" ]; then
    echo "🐍 Installing Miniconda..."
    mkdir -p ~/miniconda3
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
    bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
    rm -rf ~/miniconda3/miniconda.sh
    ~/miniconda3/bin/conda init bash
else
    echo "✅ Miniconda is already installed."
fi

# IMPORTANT: This allows the bash script to use 'conda activate'
source ~/miniconda3/etc/profile.d/conda.sh

# 4. Fix Conda ToS and Pydantic Bug in Base
echo "🔧 Configuring Conda-Forge and fixing base plugins..."
conda activate base
python -m pip install pydantic pydantic-settings pydantic-core --quiet
conda config --add channels conda-forge
conda config --set channel_priority strict

# 5. Create the Trading Environment
echo "🏗️ Creating Python 3.12 environment ('algo')..."
if conda info --envs | grep -q 'algo'; then
    echo "⚠️ Environment 'algo' already exists. Recreating it..."
    conda env remove -n algo -y
fi
conda create -n algo python=3.12 -y

# 6. Activate and Install Dependencies
echo "📥 Installing Python dependencies..."
conda activate algo
conda install pip -y
cat requirements.txt | xargs -I {} pip install --no-cache-dir {}

# 7. Setup Playwright (with Ubuntu 26.04 Bypass)
echo "🎭 Installing Playwright browsers and dependencies..."
export PLAYWRIGHT_SKIP_VALIDATE_HOST_REQUIREMENTS=true
playwright install
sudo -E playwright install-deps

# 8. Generate .env Template
echo "🔐 Checking for .env file..."
if [ ! -f ~/Algo/.env ]; then
    echo "Creating .env template..."
    cat <<EOT >> ~/Algo/.env
MOBILE_NUM=
UPSTOX_PIN=
API_KEY=
API_SECRET=
EOT
    echo "⚠️ A blank .env file was created. You MUST edit it with your credentials before running the engine!"
else
    echo "✅ .env file already exists."
fi

echo "=================================================="
echo "🎉 SETUP COMPLETE!"
echo "=================================================="
echo "Next steps:"
echo "1. Run 'source ~/.bashrc' to refresh your terminal."
echo "2. Edit your credentials: nano ~/Algo/.env"
echo "3. Activate the environment: conda activate algo"
echo "4. Download historical data [Nifty 50]: python $PWD/loki/scripts/download_historical.py --exp"
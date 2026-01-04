#!/bin/bash
# CUDA Kernel Tutorial - Setup Script
# This script sets up the development environment

set -e

echo "========================================"
echo "CUDA Kernel Tutorial - Environment Setup"
echo "========================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check for CUDA
echo -e "\n${YELLOW}Checking CUDA installation...${NC}"
if command -v nvcc &> /dev/null; then
    CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $5}' | cut -d',' -f1)
    echo -e "${GREEN}Found CUDA $CUDA_VERSION${NC}"
else
    echo -e "${RED}CUDA not found! Please install CUDA Toolkit 11.0+${NC}"
    echo "Visit: https://developer.nvidia.com/cuda-downloads"
    exit 1
fi

# Check for GPU
echo -e "\n${YELLOW}Checking GPU...${NC}"
if command -v nvidia-smi &> /dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
    echo -e "${GREEN}Found GPU: $GPU_NAME${NC}"
else
    echo -e "${RED}nvidia-smi not found! Is the NVIDIA driver installed?${NC}"
    exit 1
fi

# Check for CMake
echo -e "\n${YELLOW}Checking CMake...${NC}"
if command -v cmake &> /dev/null; then
    CMAKE_VERSION=$(cmake --version | head -1 | awk '{print $3}')
    echo -e "${GREEN}Found CMake $CMAKE_VERSION${NC}"
else
    echo -e "${RED}CMake not found! Please install CMake 3.18+${NC}"
    exit 1
fi

# Check for Python
echo -e "\n${YELLOW}Checking Python...${NC}"
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version | awk '{print $2}')
    echo -e "${GREEN}Found Python $PYTHON_VERSION${NC}"
else
    echo -e "${RED}Python 3 not found! Please install Python 3.8+${NC}"
    exit 1
fi

# Create virtual environment
echo -e "\n${YELLOW}Setting up Python virtual environment...${NC}"
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo -e "${GREEN}Created virtual environment in ./venv${NC}"
else
    echo "Virtual environment already exists"
fi

# Activate and install dependencies
source venv/bin/activate
echo -e "\n${YELLOW}Installing Python dependencies...${NC}"
pip install --upgrade pip > /dev/null
pip install -r requirements.txt

# Check for PyTorch with CUDA
echo -e "\n${YELLOW}Checking PyTorch CUDA support...${NC}"
python3 -c "import torch; print(f'PyTorch {torch.__version__} with CUDA {torch.version.cuda}')" 2>/dev/null || {
    echo -e "${YELLOW}Installing PyTorch with CUDA support...${NC}"
    pip install torch --index-url https://download.pytorch.org/whl/cu121
}

# Check for Triton
echo -e "\n${YELLOW}Checking Triton...${NC}"
python3 -c "import triton; print(f'Triton {triton.__version__}')" 2>/dev/null || {
    echo -e "${YELLOW}Installing Triton...${NC}"
    pip install triton
}

# Build the tutorial examples
echo -e "\n${YELLOW}Building CUDA examples...${NC}"
mkdir -p build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
cd ..

echo -e "\n${GREEN}========================================"
echo "Setup complete!"
echo "========================================${NC}"
echo ""
echo "To activate the environment:"
echo "  source venv/bin/activate"
echo ""
echo "To run examples:"
echo "  cd build"
echo "  ./chapters/01_introduction/examples/01_hello_cuda/hello_cuda"
echo ""
echo "To run Python examples:"
echo "  cd chapters/07_triton/examples/01_vector_add"
echo "  python vector_add.py"
echo ""
echo "Happy kernel writing!"

Links
- [course drive](https://drive.google.com/drive/folders/1ayzzvvr_w8l16cUQK2dmBhv4aUx8Gsf-)
This repository contains materials and projects for the MLOps course, including data processing pipelines, machine learning models, and deployment examples.

## Prerequisites

### VS Code Installation

1. **Download VS Code**
   - Visit [https://code.visualstudio.com/](https://code.visualstudio.com/)
   - Download and install vs code

2. **Install Essential Extensions**
   - [jupter-notebook extension](https://marketplace.visualstudio.com/items?itemName=ms-toolsai.jupyter)

### UV Installation

UV is a fast Python package installer and resolver. Install it using one of these methods:

```bash
# Using Homebrew (recommended for macOS)
brew install uv

# Or using curl
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## Setup Instructions

### 1. Clone the Repository

```bash
git clone <repo_url>
```

### 2. Sync Dependencies with UV

```bash
# Install all dependencies from pyproject.toml
uv sync
```

This command will:
- Create a virtual environment (if it doesn't exist)
- Install all project dependencies
- Install the project in editable mode

### 3. Activate the Virtual Environment

```bash
# Activate the virtual environment
source .venv/bin/activate

# Or alternatively, you can use uv to run commands in the virtual environment
cd <directory_name> && python3 (or) python <script_name>.py
```

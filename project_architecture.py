# making the project structure using python coding
from pathlib import Path

# CWD
root = Path(".")

# Folders to create
folders = [
    "app/api",
    "app/core",
    "app/rag",
    "app/services",
    "data",
    "templates",
    "static",
    "uploads",
    "tests"
]

# Files to create
files = [
    "app/main.py",
    "app/api/routes.py",
    "app/core/config.py",
    "app/core/logging.py",
    "app/core/__init__.py",

    "ingest_sample_kb.py",
    "requirements.txt",
    "run.py",
    ".env",

    "Dockerfile",
    "steps.md",
    "app/__init__.py",
]

# create folders
for folder in folders:
    (root/folder).mkdir(parents=True, exist_ok=True)

for file in files:
    (root/file).touch(exist_ok=True)

print("Project structure is created.")

# exist_ok=True
# *"If the folder/file already exists, don't crash or throw an error—just ignore it and move on."*

# parents=True
# In Python's pathlib library, parents=True tells the .mkdir() method to automatically create any missing parent (intermediate) directories in the path if they don't already exist.
# It is the Python equivalent of the Unix command mkdir - p.

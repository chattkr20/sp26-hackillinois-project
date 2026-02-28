# Create venv if it doesn't exist
if (!(Test-Path ".venv")) {
    python -m venv .venv
}

# Activate venv
.venv\Scripts\Activate.ps1

# Install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt

# Start server
uvicorn main:app --reload
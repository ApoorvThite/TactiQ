"""Launch the TactiQ Streamlit dashboard."""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

subprocess.run([
    sys.executable, '-m', 'streamlit', 'run',
    str(ROOT / 'src' / 'dashboard' / 'Home.py'),
    '--server.port=8501',
    '--server.headless=false',
], check=True)

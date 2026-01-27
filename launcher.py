import subprocess
import sys
import webbrowser
import time
from pathlib import Path

def main():
    base_dir = Path(__file__).resolve().parent
    app_path = base_dir / "app.py"

    cmd = [
        sys.executable, "-m", "streamlit", "run", str(app_path),
        "--server.address", "127.0.0.1",
        "--server.port", "8501",
        "--server.headless", "true",
    ]

    p = subprocess.Popen(cmd, cwd=str(base_dir))
    time.sleep(1.5)
    webbrowser.open("http://127.0.0.1:8501")
    p.wait()

if __name__ == "__main__":
    main()

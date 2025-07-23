import subprocess
subprocess.Popen(["uvicorn", "app.backend:app", "--host", "0.0.0.0", "--port", "8000"])
import time; time.sleep(2)
import os
os.system("streamlit run frontend/interface.py")
"""
Environment and Hardware Inspection Script
Adaptive LLM Routing Project
"""
import os
import sys
import platform
import json
import psutil

def check_environment():
    report = {}

    # OS and Python
    report["os"] = {
        "platform": platform.platform(),
        "system": platform.system(),
        "release": platform.release(),
        "version": platform.version(),
        "architecture": platform.machine()
    }
    report["python"] = {
        "version": sys.version,
        "executable": sys.executable,
        "path": sys.path
    }

    # CPU info
    report["cpu"] = {
        "physical_cores": psutil.cpu_count(logical=False),
        "logical_cores": psutil.cpu_count(logical=True),
        "current_utilization_pct": psutil.cpu_percent(interval=0.5)
    }

    # RAM info
    mem = psutil.virtual_memory()
    report["ram"] = {
        "total_gb": round(mem.total / (1024**3), 2),
        "available_gb": round(mem.available / (1024**3), 2),
        "used_gb": round(mem.used / (1024**3), 2),
        "percent_used": mem.percent
    }

    # GPU info (via pynvml, torch or subprocess nvidia-smi)
    gpu_info = {"available": False, "device_name": None, "vram_total_gb": None, "vram_free_gb": None}
    try:
        import torch
        if torch.cuda.is_available():
            gpu_info["available"] = True
            gpu_info["device_name"] = torch.cuda.get_device_name(0)
            gpu_info["device_count"] = torch.cuda.device_count()
            gpu_info["vram_total_gb"] = round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2)
    except Exception:
        pass
    report["gpu"] = gpu_info

    # Ollama status
    ollama_info = {"installed": False, "running": False, "models": []}
    try:
        import httpx
        try:
            r = httpx.get("http://127.0.0.1:11434/api/tags", timeout=2.0)
            if r.status_code == 200:
                ollama_info["installed"] = True
                ollama_info["running"] = True
                data = r.json()
                ollama_info["models"] = [m.get("name") for m in data.get("models", [])]
        except Exception as e:
            ollama_info["running"] = False
            ollama_info["error"] = str(e)
    except Exception as e:
        ollama_info["error"] = str(e)
    report["ollama"] = ollama_info

    # Key packages
    pkg_map = {
        "pandas": "pandas",
        "numpy": "numpy",
        "scikit-learn": "sklearn",
        "fastapi": "fastapi",
        "uvicorn": "uvicorn",
        "streamlit": "streamlit",
        "plotly": "plotly",
        "seaborn": "seaborn",
        "ollama": "ollama",
        "psutil": "psutil"
    }
    packages = {}
    for pkg_display, mod_name in pkg_map.items():
        try:
            mod = __import__(mod_name)
            packages[pkg_display] = getattr(mod, "__version__", "installed")
        except ImportError:
            packages[pkg_display] = "not_installed"
    report["packages"] = packages

    return report

if __name__ == "__main__":
    rep = check_environment()
    print(json.dumps(rep, indent=2))
    os.makedirs("results", exist_ok=True)
    with open("results/environment_info.json", "w") as f:
        json.dump(rep, f, indent=2)
    print("\n[OK] Environment report saved to results/environment_info.json")

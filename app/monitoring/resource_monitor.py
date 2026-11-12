"""
Hardware Resource Monitor
Tracks Host CPU, RAM utilization, and GPU/VRAM utilization where available.
"""

import psutil
from typing import Dict, Any, Optional

class ResourceMonitor:
    """Profiles hardware utilization metrics during model execution."""

    @staticmethod
    def get_current_snapshot() -> Dict[str, Any]:
        """Captures instantaneous CPU, RAM, and GPU readings."""
        mem = psutil.virtual_memory()
        cpu_pct = psutil.cpu_percent(interval=None)

        snapshot = {
            "cpu_utilization_pct": float(cpu_pct),
            "ram_used_mb": round(mem.used / (1024 * 1024), 2),
            "ram_available_mb": round(mem.available / (1024 * 1024), 2),
            "ram_percent": float(mem.percent),
            "gpu_available": False,
            "gpu_utilization_pct": None,
            "vram_used_mb": None,
            "vram_total_mb": None
        }

        # Attempt to probe GPU metrics safely via PyTorch or pynvml
        try:
            import torch
            if torch.cuda.is_available():
                snapshot["gpu_available"] = True
                snapshot["gpu_name"] = torch.cuda.get_device_name(0)
                mem_alloc = torch.cuda.memory_allocated(0) / (1024 * 1024)
                mem_total = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)
                snapshot["vram_used_mb"] = round(mem_alloc, 2)
                snapshot["vram_total_mb"] = round(mem_total, 2)
        except Exception:
            pass

        return snapshot

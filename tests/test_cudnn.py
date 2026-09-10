import ctypes
import os
import sys
from pathlib import Path

cudnn_path = str(Path(sys.prefix) / 'Lib/site-packages/nvidia/cudnn/bin')
dlls = [
    'cudnn64_9.dll',
    'cudnn_adv64_9.dll',
    'cudnn_cnn64_9.dll',
    'cudnn_engines_precompiled64_9.dll',
    'cudnn_engines_runtime_compiled64_9.dll',
    'cudnn_engines_tensor_ir64_9.dll',
    'cudnn_ext64_9.dll',
    'cudnn_graph64_9.dll',
    'cudnn_heuristic64_9.dll',
    'cudnn_ops64_9.dll',
]

for dll_name in dlls:
    dll_path = os.path.join(cudnn_path, dll_name)
    try:
        if os.path.exists(dll_path):
            ctypes.WinDLL(dll_path)
            print(f"✅ {dll_name} loaded successfully")
        else:
            print(f"❌ {dll_name} not found")
    except Exception as e:
        print(f"❌ {dll_name} failed to load: {e}")

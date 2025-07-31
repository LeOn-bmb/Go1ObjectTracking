import os
import subprocess

# Modellverzeichnis
model_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../models'))   # Pfad ggf. anpassen
trtexec_path = '/usr/src/tensorrt/bin/trtexec'  
use_fp16 = True

print(f"🚀 Starte Konvertierung von ONNX nach TensorRT-Engines im Verzeichnis: {model_dir}\n")

for filename in os.listdir(model_dir):
    if filename.endswith("_int32.onnx"):    # Datei(en), die mit "_int32.onnx" enden konvertieren
        onnx_path = os.path.join(model_dir, filename)
        engine_path = onnx_path.replace("_int32.onnx", ".engine") 

        if os.path.exists(engine_path):
            print(f"⚠️  {engine_path} existiert bereits – überspringe.")
            continue

        print(f"🔧 Erstelle .engine für: {filename}")
        cmd = [
            trtexec_path,
            f'--onnx={onnx_path}',
            f'--saveEngine={engine_path}',
            '--workspace=2048',
            '--verbose'
        ]
        if use_fp16:
            cmd.append('--fp16')

        try:
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
            if result.returncode == 0:
                print(f"✅ Engine erstellt: {engine_path}\n")
            else:
                print(f"❌ Fehler bei {filename}:\n{result.stderr}\n")
        except Exception as e:
            print(f"❌ Unerwarteter Fehler: {e}\n")
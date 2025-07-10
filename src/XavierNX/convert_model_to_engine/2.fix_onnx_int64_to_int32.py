import onnx
import onnx_graphsurgeon as gs
import numpy as np
import os

model_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../models"))

for file in os.listdir(model_dir):
    if file.endswith(".onnx"):
        input_path = os.path.join(model_dir, file)
        output_path = input_path.replace(".onnx", "_int32.onnx")
        print(f"🔧 Fixing {file} -> {os.path.basename(output_path)}")

        graph = gs.import_onnx(onnx.load(input_path))

        for inp in graph.inputs:
            if inp.dtype == np.int64:
                print(f"✅ Input {inp.name} -> int32")
                inp.dtype = np.int32
        for out in graph.outputs:
            if out.dtype == np.int64:
                print(f"✅ Output {out.name} -> int32")
                out.dtype = np.int32

        for node in graph.nodes:
            for idx, inp in enumerate(node.inputs):
                if isinstance(inp, gs.Constant) and inp.values.dtype == np.int64:
                    print(f"✅ Constant {inp.name} -> int32")
                    inp.values = inp.values.astype(np.int32)
                elif isinstance(inp, gs.Variable) and inp.dtype == np.int64:
                    print(f"✅ Node Input {inp.name} -> int32")
                    inp.dtype = np.int32

            for out in node.outputs:
                if out.dtype == np.int64:
                    print(f"✅ Node Output {out.name} -> int32")
                    out.dtype = np.int32

        graph.cleanup().toposort()

        onnx.save(gs.export_onnx(graph), output_path)
        print(f"✅ Saved: {output_path}\n")
import onnx
import glob
import os
import json

target_dir = r"N:\models\onnx\nemo\parakeet-tdt-0.6b-v2-onnx-tfjs4\onnx"
config_path = r"N:\models\onnx\nemo\parakeet-tdt-0.6b-v2-onnx-tfjs4\config.json"

print("Fixing ONNX files...")
for file in glob.glob(os.path.join(target_dir, "*.onnx")):
    print(f"Inspecting {file}...")
    model = onnx.load(file, load_external_data=False)
    
    modified = False
    for tensor in model.graph.initializer:
        if tensor.HasField("data_location") and tensor.data_location == onnx.TensorProto.EXTERNAL:
            for ext_entry in tensor.external_data:
                if ext_entry.key == "location":
                    old_name = ext_entry.value
                    
                    # Compute canonical data name
                    expected_data_name = os.path.basename(file) + "_data"
                    
                    if old_name != expected_data_name:
                        print(f"  Updating reference from '{old_name}' -> '{expected_data_name}'")
                        ext_entry.value = expected_data_name
                        modified = True
                        
    if modified:
        onnx.save(model, file)
        print(f"Saved {file} successfully.")
    else:
        print(f"No changes needed for {file}.")

print("\nUpdating config.json...")
with open(config_path, "r") as f:
    config = json.load(f)

if "transformers.js_config" not in config:
    config["transformers.js_config"] = {}

# We only have external data for the fp32 encoder_model.onnx
# The fp16 and int8 files are small enough to be monolithic
config["transformers.js_config"]["use_external_data_format"] = {
    "encoder_model.onnx": 1
}

with open(config_path, "w") as f:
    json.dump(config, f, indent=2)
print("Updated config.json successfully.")

import numpy as np
import os


# Estimate number of parameteres
PATH_TO_ZOO = "../depthai-model-zoo"

for fname in os.listdir(os.path.join(PATH_TO_ZOO, "models")):
    bin_path = os.path.join(PATH_TO_ZOO, "models", fname, f"{fname}.bin")
    mparams = np.fromfile(bin_path, dtype=np.float16).shape[0] / 1000000
    print(f"[MParams] {fname}: {mparams}")





# Old code using model zoo
"""
zip_path = blobconverter.from_zoo(name=args.model_name,
                                    zoo_type=args.zoo_type,
                                    shaves=args.shaves,
                                    use_cache = False,
                                    download_ir = True)
bin_path = f"{args.model_name}.bin"
with zipfile.ZipFile(zip_path, "r") as zip_ref:
    zip_ref.extract(bin_path)

print(f"MParams: {np.fromfile(bin_path, dtype=np.float16).shape[0] / 1000000}")
os.remove(bin_path)
"""
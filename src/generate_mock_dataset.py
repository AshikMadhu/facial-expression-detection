import os
import numpy as np

# Resolve path mappings
src_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.abspath(os.path.join(src_dir, "..", "data"))
os.makedirs(data_dir, exist_ok=True)
csv_path = os.path.join(data_dir, "fer2013.csv")

print(f"Generating mock FER2013 dataset in: {csv_path}...")

# Target record sizes
total = 35887
train_cnt = 28709
val_cnt = 3589
test_cnt = 3589

# Generate random emotions and split usage tags
np.random.seed(42)
emotions = np.random.randint(0, 7, total)
usages = (
    ["Training"] * train_cnt +
    ["PublicTest"] * val_cnt +
    ["PrivateTest"] * test_cnt
)

# Flattened pixel sequence representing a neutral gray block (2304 integers)
pixel_val = " ".join(["127"] * 2304)

# Write to CSV in a streaming format to optimize memory footprint
with open(csv_path, "w") as f:
    f.write("emotion,pixels,Usage\n")
    for i in range(total):
        f.write(f"{emotions[i]},{pixel_val},{usages[i]}\n")
        if (i + 1) % 5000 == 0:
            print(f"Written {i + 1} / {total} rows...")
            
print("Mock FER2013 dataset successfully generated!")

import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import os
from pathlib import Path

# Set up paths - point to your converted .mat files
data_path = Path("/Users/jasonwang/Desktop/Wearable/EMG_Project/grabmyo_1.1.0/Output_BM/Session1_converted")  # Start with Session 1
print(f"Looking for data in: {data_path.absolute()}")

# Step 1: See what files you have
mat_files = list(data_path.glob("*.mat"))
print(f"\nFound {len(mat_files)} .mat files:")
for file in mat_files[:5]:  # Show first 5 files
    print(f"  - {file.name}")

# Step 2: Load and examine one file
if mat_files:
    # Load the first file
    sample_file = mat_files[0]
    print(f"\nExamining file: {sample_file.name}")
    
    # Load the .mat file
    data = sio.loadmat(sample_file)
    
    # See what variables are in the file
    print("\nVariables in .mat file:")
    for key, value in data.items():
        if not key.startswith('__'):  # Skip metadata
            print(f"  {key}: {type(value)} - Shape: {getattr(value, 'shape', 'No shape')}")

    # Step 3: Let's examine the EMG data structure
    # Try to find the main data variables
    for key, value in data.items():
        if not key.startswith('__'):
            if isinstance(value, np.ndarray):
                print(f"\n{key} details:")
                print(f"  Shape: {value.shape}")
                print(f"  Data type: {value.dtype}")
                
                # Handle object arrays (which might contain structured data)
                if value.dtype == 'object':
                    print(f"  This is an object array - likely contains structured data")
                    if value.size > 0:
                        print(f"  First element type: {type(value.flat[0])}")
                        first_elem = value.flat[0]
                        if hasattr(first_elem, 'shape'):
                            print(f"  First element shape: {first_elem.shape}")
                        if isinstance(first_elem, np.ndarray) and first_elem.size > 0:
                            print(f"  First element sample: {first_elem.flat[:3]}")
                else:
                    # Handle regular numeric arrays
                    if value.size > 0:
                        print(f"  Min/Max: {value.min():.3f} / {value.max():.3f}")
                        print(f"  Sample values: {value.flat[:5]}")

print("\n" + "="*50)
print("NEXT: Tell me what variables you see!")
print("Common ones should be:")
print("- EMG signals (maybe 'emg', 'data', or 'signal')")
print("- Labels/gestures (maybe 'labels', 'gestures', 'stimulus')")
print("- Sampling rate info")
print("="*50)
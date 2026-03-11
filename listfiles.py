import os, glob

# Check multiple likely locations
for folder in [".", "./outputs", "./results", "./data", "./saved", "./experiment_outputs"]:
    if os.path.exists(folder):
        files = glob.glob(os.path.join(folder, "**", "*.*"), recursive=True)
        if files:
            print(f"\n=== {folder} ===")
            for f in sorted(files):
                size_kb = os.path.getsize(f) / 1024
                print(f"  {f}  ({size_kb:.1f} KB)")

# Also check current directory directly
print("\n=== Current directory ===")
for f in sorted(os.listdir(".")):
    print(f"  {f}")
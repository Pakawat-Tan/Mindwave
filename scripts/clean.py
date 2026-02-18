import shutil
from pathlib import Path

print("🧹 Cleaning Python cache...")

root = Path(".")
pycache_dirs = list(root.rglob("__pycache__"))
pyc_files = list(root.rglob("*.pyc"))

print(f"Found {len(pycache_dirs)} __pycache__ directories")
print(f"Found {len(pyc_files)} .pyc files")

if not pycache_dirs and not pyc_files:
    print("Nothing to clean.")
    exit()

confirm = input("Proceed? (y/n): ")

if confirm.lower() == "y":
    for d in pycache_dirs:
        shutil.rmtree(d, ignore_errors=True)

    for f in pyc_files:
        f.unlink(missing_ok=True)

    print("✅ Done.")
else:
    print("Cancelled.")

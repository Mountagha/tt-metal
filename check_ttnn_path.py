#!/usr/bin/env python3
import sys

print("Python path:")
for p in sys.path:
    print(f"  {p}")

print("\nImporting ttnn...")
import ttnn

print(f"ttnn module location: {ttnn.__file__}")
print(f"ttnn version: {ttnn.__version__ if hasattr(ttnn, '__version__') else 'N/A'}")

# Check if it's the local build
if "/root/workspace/tt-metal" in ttnn.__file__:
    print("\n✓ Using LOCAL build from workspace")
else:
    print("\n✗ Using INSTALLED version (pip)")

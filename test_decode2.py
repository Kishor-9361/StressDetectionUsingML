import audioread
import sys

print("Python version:", sys.version)
print("Audioread module path:", audioread.__file__)
try:
    # Try opening a dummy file to see how it resolves decoders
    from audioread import decoders
    print("Decoders in audioread:")
    for d in decoders():
        print("  -", d)
except Exception as e:
    print("Error:", e)

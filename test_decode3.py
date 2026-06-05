import audioread
try:
    print("Available backends:", [b.__module__ for b in audioread.available_backends()])
except Exception as e:
    print("Error:", e)

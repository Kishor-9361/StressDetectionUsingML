import audioread
import traceback

print("Checking audioread decoders...")
try:
    decoders = []
    # Let's inspect the available backends in audioread
    import audioread.ffdec
    decoders.append(("ffdec", audioread.ffdec.available()))
    import audioread.gstdec
    decoders.append(("gstdec", audioread.gstdec.available()))
    import audioread.maddec
    decoders.append(("maddec", audioread.maddec.available()))
    
    # On Windows, audioread has rawdec or standard backends
    try:
        import audioread.macca
        decoders.append(("macca", audioread.macca.available()))
    except ImportError:
        pass
        
    print("Decoders available status:")
    for name, avail in decoders:
        print(f"  - {name}: {avail}")
except Exception as e:
    print("Error:")
    traceback.print_exc()

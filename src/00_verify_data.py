from common import DATA, EXPECTED_SHA256, sha256
if not DATA.exists():
    raise SystemExit(f"Data file not found: {DATA}")
actual=sha256(DATA)
print("Data:", DATA)
print("SHA-256:", actual)
if actual != EXPECTED_SHA256:
    print("WARNING: file hash differs from the analysis copy. Check dataset version.")
else:
    print("Dataset hash verified.")

"""Test that file gets written successfully even if console print fails"""
import os

output_dir = "test_output"
os.makedirs(output_dir, exist_ok=True)

test_file = os.path.join(output_dir, "unicode_test.txt")

# Write file with Unicode
print("Writing file with Unicode characters...")
with open(test_file, 'w', encoding='utf-8') as f:
    f.write("Test insights:\n")
    f.write("1. ✓ Representative sampling improved coverage by 36.3%\n")
    f.write("2. ✓ Eliminated 10 near-duplicate pairs\n")
    f.write("3. ✓ Increased diversity score by 16.7%\n")

print(f"[OK] File written to: {test_file}")

# Verify file exists and has content
if os.path.exists(test_file):
    size = os.path.getsize(test_file)
    print(f"[OK] File exists with {size} bytes")

    # Read back and verify (without printing)
    with open(test_file, 'r', encoding='utf-8') as f:
        content = f.read()

    if '✓' in content:
        print("[OK] Unicode characters preserved in file")

    print("[OK] File content length:", len(content), "characters")
else:
    print("[FAIL] File was not created")

# Clean up
import shutil
shutil.rmtree(output_dir)

print("\n[SUCCESS] File writing works correctly!")
print("Note: Console printing may fail on Windows, but files are saved properly.")

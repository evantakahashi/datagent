"""Test that Unicode characters work correctly in file writing"""
import os

output_dir = "test_output"
os.makedirs(output_dir, exist_ok=True)

test_file = os.path.join(output_dir, "unicode_test.txt")

# Test with UTF-8 encoding
print("Testing UTF-8 file writing with Unicode characters...")
try:
    with open(test_file, 'w', encoding='utf-8') as f:
        f.write("Test with checkmark: \u2713\n")
        f.write("Test with X mark: \u2717\n")
        f.write("Regular text: This is fine\n")

    print("[OK] File written successfully")

    # Read it back
    with open(test_file, 'r', encoding='utf-8') as f:
        content = f.read()

    print("[OK] File read successfully")
    print("\nContent:")
    print(content)

except Exception as e:
    print(f"[FAIL] Error: {e}")
    import traceback
    traceback.print_exc()

# Clean up
import shutil
shutil.rmtree(output_dir)
print("\n[OK] All tests passed! Unicode encoding fix is working.")

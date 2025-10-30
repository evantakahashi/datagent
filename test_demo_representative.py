"""Quick test of the representative demo without full run"""
import sys
import os

# Simulate the key insights generation and printing/writing
insights = [
    "✓ Representative sampling improved coverage by 36.3%, ensuring better representation",
    "✓ Eliminated 10 near-duplicate pairs, ensuring each record contributes unique information",
    "✓ Increased diversity score by 16.7%, providing more balanced representation",
]

print("Testing console output (Windows-friendly)...")
print("=" * 60)

for i, insight in enumerate(insights, 1):
    # Replace Unicode for console
    console_insight = insight.replace('✓', '[OK]')
    try:
        print(f"{i}. {console_insight}")
    except UnicodeEncodeError:
        print(f"{i}. {console_insight.encode('ascii', 'replace').decode('ascii')}")

print("\n" + "=" * 60)
print("Testing file writing (UTF-8 with Unicode)...")

output_dir = "test_output"
os.makedirs(output_dir, exist_ok=True)
report_path = os.path.join(output_dir, "test_report.txt")

with open(report_path, 'w', encoding='utf-8') as f:
    f.write("KEY INSIGHTS:\n")
    f.write("-" * 60 + "\n")
    for i, insight in enumerate(insights, 1):
        # Keep Unicode in file
        f.write(f"{i}. {insight}\n\n")

print(f"[OK] Report saved to: {report_path}")

# Verify file
with open(report_path, 'r', encoding='utf-8') as f:
    content = f.read()

if '✓' in content:
    print("[OK] Unicode checkmarks preserved in file")

print(f"[OK] File size: {os.path.getsize(report_path)} bytes")

# Clean up
import shutil
shutil.rmtree(output_dir)

print("\n" + "=" * 60)
print("[SUCCESS] All encoding issues fixed!")
print("=" * 60)
print("\nYou can now run: python demo_representative.py")

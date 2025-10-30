"""Quick test to verify all imports work before running demo"""
import sys

print("Testing imports...")
print("-" * 50)

try:
    from src.generators import PatientRecordGenerator, IterativeGenerator
    print("[OK] Generators imported successfully")
except Exception as e:
    print(f"[FAIL] Generators import failed: {e}")
    sys.exit(1)

try:
    from src.models import PatientRecord, Demographics, Sex, Ethnicity
    print("[OK] Models imported successfully")
except Exception as e:
    print(f"[FAIL] Models import failed: {e}")
    sys.exit(1)

try:
    from src.evaluators import (
        EmbeddingAnalyzer,
        DistributionMetrics,
        ConceptEntropyCalculator,
        DiversityVisualizer,
        RedundancyChecker,
        ActiveDataSelector,
        CoverageAnalyzer
    )
    print("[OK] Evaluators imported successfully")
except Exception as e:
    print(f"[FAIL] Evaluators import failed: {e}")
    sys.exit(1)

try:
    from src.utils import ReportGenerator
    print("[OK] Utils imported successfully")
except Exception as e:
    print(f"[FAIL] Utils import failed: {e}")
    sys.exit(1)

# Quick generation test
try:
    print("\nTesting patient record generation...")
    gen = PatientRecordGenerator(seed=42)
    record = gen.generate_record()
    print(f"[OK] Generated test record: {record.record_id}")
    print(f"  Demographics: {record.demographics.age}y, {record.demographics.sex.value}, {record.demographics.ethnicity.value}")
    print(f"  Conditions: {len(record.conditions)}")
    print(f"  Medications: {len(record.medications)}")
    print(f"  Labs: {len(record.lab_results)}")
except Exception as e:
    print(f"[FAIL] Generation test failed: {e}")
    sys.exit(1)

print("\n" + "=" * 50)
print("ALL TESTS PASSED!")
print("=" * 50)
print("\nYou can now run:")
print("  python demo.py                    # Full diversity evaluation")
print("  python demo_representative.py     # Compare random vs representative")
print("  python examples/simple_generation.py")
print("  python examples/evaluate_diversity.py")
print("  python examples/representative_sampling.py")

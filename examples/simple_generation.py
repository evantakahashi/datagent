"""
Simple example: Generate synthetic patient records

This shows the most basic usage - just generating synthetic patient records
without evaluation metrics.
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.generators import PatientRecordGenerator


def main():
    print("Simple Synthetic Patient Record Generation Example")
    print("=" * 60)
    print()

    # Create generator
    generator = PatientRecordGenerator(seed=42)

    # Generate a single record
    print("Generating a single patient record...")
    record = generator.generate_record()

    print("\n" + "=" * 60)
    print("GENERATED PATIENT RECORD")
    print("=" * 60)
    print(record.to_text())
    print()

    # Generate multiple records with diversity constraints
    print("\nGenerating 10 diverse patient records...")
    records = generator.generate_batch(n=10, diversity_constraints=True)

    print(f"\nGenerated {len(records)} records:")
    for i, rec in enumerate(records, 1):
        print(f"\n{i}. {rec.record_id}")
        print(f"   Age: {rec.demographics.age}, Sex: {rec.demographics.sex.value}")
        print(f"   Ethnicity: {rec.demographics.ethnicity.value}")
        print(f"   Conditions: {', '.join([c.name for c in rec.conditions])}")
        print(f"   Medications: {len(rec.medications)}")
        print(f"   Lab Tests: {len(rec.lab_results)}")

    # Generate with specific constraints
    print("\n" + "=" * 60)
    print("Generating record with constraints...")
    print("Constraints: Female, Age 65")

    constrained_record = generator.generate_record(constraints={'sex': 'female', 'age': 65})

    print(f"\nGenerated: {constrained_record.record_id}")
    print(f"Age: {constrained_record.demographics.age}")
    print(f"Sex: {constrained_record.demographics.sex.value}")
    print(f"Conditions: {', '.join([c.name for c in constrained_record.conditions])}")

    print("\nDone!")


if __name__ == "__main__":
    main()

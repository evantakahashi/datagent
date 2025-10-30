"""Test that report generator handles None embedding_metrics correctly"""
from src.utils import ReportGenerator
from datetime import datetime

# Test with None embedding_metrics
results = {
    'dataset_info': {
        'n_real': 50,
        'n_synthetic': 50,
        'timestamp': datetime.now().isoformat(),
    },
    'embedding_metrics': None,  # This should not crash
    'distribution_metrics': {
        'fid_score': 45.2,
        'kl_div_age_groups': 0.023,
        'kl_div_sex': 0.045,
        'kl_div_average': 0.089,
    },
    'entropy_comparison': {
        'real_diversity_score': 0.823,
        'synthetic_diversity_score': 0.816,
        'diversity_score_diff': 0.007,
        'entropy_differences': {
            'age_groups': {'real': 0.89, 'synthetic': 0.88, 'difference': 0.01},
            'sex': {'real': 0.95, 'synthetic': 0.92, 'difference': 0.03},
        }
    },
}

print("Testing report generation with None embedding_metrics...")
report_generator = ReportGenerator()

try:
    report_text = report_generator.generate_text_report(results, None)
    print("[OK] Report generated successfully without embeddings")
    print("\nReport preview:")
    print("-" * 50)
    print(report_text[:500])
    print("...")
except Exception as e:
    print(f"[FAIL] Report generation failed: {e}")
    import traceback
    traceback.print_exc()

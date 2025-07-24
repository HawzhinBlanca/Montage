from montage.cli.run_pipeline import extract_audio_rms
import json
import os

# Create output directory if it doesn't exist
os.makedirs('qa_test_results', exist_ok=True)

# Extract audio RMS energy from the 1-minute test video
rms_levels = extract_audio_rms('/tmp/test_1min.mp4')

# Save results
with open('qa_test_results/audio_energy.json', 'w') as f:
    json.dump({
        'sample_count': len(rms_levels),
        'average_energy': sum(rms_levels) / len(rms_levels) if rms_levels else 0,
        'max_energy': max(rms_levels) if rms_levels else 0,
        'min_energy': min(rms_levels) if rms_levels else 0,
        'first_10_samples': rms_levels[:10]
    }, f, indent=2)

print(f"Audio RMS extracted: {len(rms_levels)} samples")
print(f"Average energy: {sum(rms_levels) / len(rms_levels) if rms_levels else 0:.3f}")
print(f"Max energy: {max(rms_levels) if rms_levels else 0:.3f}")
print(f"Min energy: {min(rms_levels) if rms_levels else 0:.3f}")
print(f"First 10 samples: {rms_levels[:10]}")
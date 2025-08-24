import numpy as np
import scipy.io as sio
from pathlib import Path

# Define gesture mapping based on GestureList.JPG
GESTURE_LABELS = {
    1: "Lateral prehension (LP)",
    2: "Thumb adduction (TA)", 
    3: "Thumb and little finger opposition (TLFO)",
    4: "Thumb and index finger opposition (TIFO)",
    5: "Thumb and little finger extension (TLFE)",
    6: "Thumb and index finger extension (TIFE)",
    7: "Index and middle finger extension (IMFE)",
    8: "Little finger extension (LFE)",
    9: "Index finger extension (IFE)",
    10: "Thumb extension (TE)",
    11: "Wrist flexion (WF)",
    12: "Wrist extension (WE)",
    13: "Forearm pronation (FP)",
    14: "Forearm supination (FS)",
    15: "Hand open (HO)",
    16: "Hand close (HC)",
    17: "Rest (no gesture)"
}

def analyze_emg_data():
    # Load sample data
    data_path = Path("/Users/jasonwang/Desktop/Wearable/EMG_Project/grabmyo_1.1.0/Output_BM/Session1_converted")
    sample_file = data_path / "session1_participant1.mat"
    data = sio.loadmat(sample_file)
    
    print("=== GRABMYO EMG DATABASE ANALYSIS ===\n")
    
    # Extract data arrays
    forearm_data = data['DATA_FOREARM'] 
    wrist_data = data['DATA_WRIST']
    
    print("📊 DATA STRUCTURE:")
    print(f"  • Dataset: GRABMyo Database (43 participants, 3 sessions)")
    print(f"  • Sampling rate: 2048 Hz")
    print(f"  • Trial structure: {forearm_data.shape[0]} trials × {forearm_data.shape[1]} gestures")
    print(f"  • Forearm channels: {forearm_data[0,0].shape[1]} (16 electrode locations)")
    print(f"  • Wrist channels: {wrist_data[0,0].shape[1]} (12 electrode locations)")
    print(f"  • Samples per trial: {forearm_data[0,0].shape[0]} (~5 seconds at 2048Hz)")
    
    print(f"\n🎯 GESTURE CLASSES ({len(GESTURE_LABELS)} total):")
    for gesture_id, label in GESTURE_LABELS.items():
        print(f"  {gesture_id:2d}: {label}")
    
    print(f"\n📈 DATA ORGANIZATION:")
    print(f"  • Each .mat file contains one participant's data")
    print(f"  • DATA_FOREARM[trial_idx, gesture_idx] → (10240, 16) array")
    print(f"  • DATA_WRIST[trial_idx, gesture_idx] → (10240, 12) array")
    print(f"  • Total data per participant: {7 * 17 * (16 + 12) * 10240:,} EMG samples")
    
    # Sample data inspection
    print(f"\n🔬 SIGNAL CHARACTERISTICS:")
    sample_forearm = forearm_data[0, 0]  # Trial 1, Gesture 1 (LP)
    sample_wrist = wrist_data[0, 0]
    
    print(f"  • Forearm EMG range: [{sample_forearm.min():.3f}, {sample_forearm.max():.3f}] V")
    print(f"  • Wrist EMG range: [{sample_wrist.min():.3f}, {sample_wrist.max():.3f}] V")
    
    print(f"\n✅ READY FOR MACHINE LEARNING:")
    print(f"  • Features: EMG signals from 28 channels (16 forearm + 12 wrist)")
    print(f"  • Labels: 17 gesture classes")
    print(f"  • Training data: 7 trials per gesture per participant")
    print(f"  • Multiple participants available for robust training")
    
    return {
        'forearm_channels': 16,
        'wrist_channels': 12, 
        'total_channels': 28,
        'gestures': 17,
        'trials_per_gesture': 7,
        'samples_per_trial': 10240,
        'sampling_rate': 2048,
        'gesture_labels': GESTURE_LABELS
    }

if __name__ == "__main__":
    info = analyze_emg_data()
    
    print(f"\n" + "="*60)
    print("🚀 NEXT STEPS FOR EMG GESTURE RECOGNITION:")
    print("="*60)
    print("1. Load and concatenate data from multiple participants")
    print("2. Extract time-domain and frequency-domain features") 
    print("3. Apply machine learning (SVM, Random Forest, Neural Networks)")
    print("4. Evaluate classification performance")
    print("5. Test real-time gesture recognition")
    print("="*60)
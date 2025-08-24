Directory Structure Explained
data/ - All your datasets live here

    raw/ - Original, untouched GRABMyo files. Never modify these! Think of this as your backup - if you mess up preprocessing, you can always start over

    processed/ - Cleaned data after filtering, segmentation, feature extraction. This is what you'll actually train models on

notebooks/ - Jupyter notebooks for exploration and experimentation

    Great for: data visualization, trying different approaches, creating plots for your portfolio
    Examples: 01_data_exploration.ipynb, 02_signal_visualization.ipynb, 03_model_comparison.ipynb

src/ - Reusable Python modules (the "production" code)

    preprocessing.py - Functions for filtering EMG signals, removing artifacts, segmentation
    features.py - Code to extract features like RMS, mean absolute value, frequency features
    models.py - Model training, evaluation, and prediction functions

results/ - All outputs from your experiments

    Trained model files (.pkl, .h5)
    Performance plots, confusion matrices
    Classification reports, accuracy tables

README.md - Project documentation (crucial for your portfolio!)




Example Workflow:

Load raw data → explore in notebook → create preprocessing functions in src/
Extract features → save to processed/ → build models in src/models.py
Results automatically saved to results/ → document everything in README
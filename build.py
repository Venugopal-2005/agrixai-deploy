#!/usr/bin/env python3
"""Build script that trains the ML model if pkl files are missing or invalid.

This runs during deployment so the 140MB model doesn't need to be in Git.
"""
import os
import sys

def main():
    model_dir = os.path.join(os.path.dirname(__file__), "model")
    model_path = os.path.join(model_dir, "yield_model.pkl")

    # Check if model exists and is a real file (not a Git LFS pointer)
    needs_training = False
    if not os.path.exists(model_path):
        needs_training = True
        print("Model file not found — training required.")
    elif os.path.getsize(model_path) < 1000:
        needs_training = True
        print("Model file appears to be a Git LFS pointer — retraining required.")
    else:
        print(f"Model file exists ({os.path.getsize(model_path)} bytes) — skipping training.")

    if needs_training:
        print("Training model from dataset...")
        # Import and run the training script
        from train_model_no_crop_year import main as train_main
        train_main()
        print("Model training complete.")
    else:
        print("Build complete — model is ready.")

if __name__ == "__main__":
    main()

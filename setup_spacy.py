"""
Setup script to install spaCy language model.
Run this after installing requirements.txt
"""

import subprocess
import sys

def install_spacy_model():
    """Install the English language model for spaCy."""
    print("Installing spaCy English language model...")
    try:
        subprocess.check_call([
            sys.executable, "-m", "spacy", "download", "en_core_web_sm"
        ])
        print("✓ spaCy model installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to install spaCy model: {e}")
        return False

if __name__ == "__main__":
    success = install_spacy_model()
    sys.exit(0 if success else 1)

#wrapper
import os

assert "Python 3." in os.popen("python -V").read(), "Get python 3 you fool"

os.popen("conda config --add channels conda-forge").read()

print("Installing dependencies with conda...")
os.popen(f"conda install --yes numpy scipy nilearn joblib statsmodels").read()

print("Installing prfpy and verifying dependencies with pip...")
os.popen("python -m pip install -e .").read()

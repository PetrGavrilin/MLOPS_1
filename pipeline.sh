#!/bin/bash
pip3 install scikit-learn
pip3 install pandas
#pip install joblib

python3 data_creation.py
python3 data_preprocessing.py
python3 model_preparation.py
python3 model_testing.py

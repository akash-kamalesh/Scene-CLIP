#!/bin/bash
pip install transformers huggingface_hub datasets adapters wandb sentence_transformers
pip install numpy==1.26.0 evaluate tabulate scikit-learn wandb matplotlib 
pip install spacy
pip install SceneGraphParser
pip install numpy==1.26.0
python -m spacy download en_core_web_sm
pip install ftfy
pip install easydict 
pip install open_clip_torch
pip install opencv-python
pip install word2number
pip install setuptools==69.5.1
pip install git+https://github.com/openai/CLIP.git
pip install discosg --no-deps
pip install FactualSceneGraph --no-deps
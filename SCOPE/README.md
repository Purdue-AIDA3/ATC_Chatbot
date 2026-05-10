Unified SCOPE experiment runner - run_train_all.py
====================================================
Runs training, evaluation, and/or IFEval for any combination of:
  - Domains:     atc, smcp (maritime)
  - Models:      gpt2, llama, qwen
  - Conditions:  C1, C2, C3, C4, C11 (and any others defined in the runners)
  - Tasks:       train, ifeval, both

Usage examples
--------------
# Full ATC run — all models, all conditions:
python run_train_all.py --domain atc --models gpt2 llama qwen --conditions C2 C3 C11 C4

# Maritime only, GPT-2, two conditions:
python run_train_all.py --domain smcp --models gpt2 --conditions C2 C11

# ATC Llama + Qwen + IFEval afterwards:
python run_train_all.py --domain atc --models llama qwen --conditions C2 C11 --tasks train ifeval

# IFEval only on already-trained models:
python run_train_all.py --domain atc --models gpt2 llama qwen --conditions C2 C11 --tasks ifeval

# Colab path:
python run_train_all.py --domain smcp --scope_dir /content/drive/MyDrive/ATC_Chatbot/SCOPE

# Gilbreth (default path):
python run_train_all.py --domain smcp
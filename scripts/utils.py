# utils.py
import collections
import json
import subprocess
import os
import re
import yaml
import random
import matplotlib
import sys
import numpy as np


# Instalação de dependências
def install_requirements():
    requirements_file = "requirements.txt"
    if os.path.exists(requirements_file):
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", requirements_file])
            print("[utils] Dependências instaladas com sucesso.")
        except subprocess.CalledProcessError as e:
            print(f"[utils] Erro ao instalar dependências: {e}")
            sys.exit(1)
    else:
        print(f"[utils] Arquivo {requirements_file} não encontrado.")
        sys.exit(1)

# Função auxiliar para obter input numérico validado
def get_validated_numerical_input(prompt_message, num_options):
    while True:
        try:
            user_input = input(prompt_message)
            choice = int(user_input)
            if 0 <= choice < num_options:
                return choice
            else:
                print(f"[utils] Opção inválida. Por favor, insira um número entre 0 e {num_options - 1}.")
        except ValueError:
            print("[utils] Entrada inválida. Por favor, insira um número.")
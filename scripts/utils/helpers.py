# utils.py
import subprocess
import os
import sys
import logging


# Instalação de dependências
logger = logging.getLogger(__name__)
def install_requirements():
    requirements_file = "requirements.txt"
    if os.path.exists(requirements_file):
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", requirements_file])
            logger.info("Dependências instaladas com sucesso.")
        except subprocess.CalledProcessError as e:
            logger.error(f"Erro ao instalar dependências: {e}")
            sys.exit(1)
    else:
        logger.error(f"Arquivo {requirements_file} não encontrado.")
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
                logger.warning(f"Opção inválida. Por favor, insira um número entre 0 e {num_options - 1}.")
        except ValueError:
            logger.warning("Entrada inválida. Por favor, insira um número.")
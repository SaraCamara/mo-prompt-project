import logging
import os
from datetime import datetime

def setup_logging(log_dir="logs", log_file_name="app.log", level=logging.INFO):
    """
    Configura o sistema de logging para o projeto.

    Args:
        log_dir (str): Diretório onde os arquivos de log serão salvos.
        log_file_name (str): Nome do arquivo de log principal.
        level (int): Nível mínimo de logging (e.g., logging.INFO, logging.DEBUG).
    """
    # Cria o diretório de logs se não existir
    os.makedirs(log_dir, exist_ok=True)

    # Define o formato das mensagens de log
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Configura o logger raiz
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Handler para console
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # Handler para arquivo
    file_handler = logging.FileHandler(os.path.join(log_dir, log_file_name))
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    logging.info("Sistema de logging configurado.")
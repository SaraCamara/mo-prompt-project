import logging
import os
from datetime import datetime

# Configura o sistema de logging para o projeto.
def setup_logging(log_dir="logs", log_file_name="app.log", level=logging.INFO):
    os.makedirs(log_dir, exist_ok=True)

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    file_handler = logging.FileHandler(os.path.join(log_dir, log_file_name))
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    logging.info("Sistema de logging configurado.")
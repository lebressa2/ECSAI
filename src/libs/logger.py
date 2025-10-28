# backend/logger.py
import logging
import sys

def setup_logger(name: str = "ECSAI", level=logging.INFO) -> logging.Logger:
    """Cria um logger configurado para o projeto inteiro."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # evita duplicar handlers ao importar várias vezes

    logger.setLevel(level)

    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%H:%M:%S"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    # Opcional: log para arquivo
    # file_handler = logging.FileHandler("logs/ecsa.log")
    # file_handler.setFormatter(formatter)
    # logger.addHandler(file_handler)

    return logger

# Instância global padrão
logger = setup_logger()

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path




def get_logger(name: str = "ecg_logger", log_file: Path = None, level=logging.INFO):
    """
    Creates a logger with console and rotating file handler support.




    Args:
        name (str): Logger name.
        log_file (Path): Optional path to log file.
        level (int): Logging level (e.g., logging.INFO, logging.DEBUG).
   
    Returns:
        Logger object.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False




    #  ====duplicate handlers
    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setLevel(level)
        formatter = logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s", "%H:%M:%S")
        ch.setFormatter(formatter)
        logger.addHandler(ch)




        # File Handler 
        if log_file:
            log_file = Path(log_file)
            log_file.parent.mkdir(parents=True, exist_ok=True)  # Ensure dir exists
            fh = RotatingFileHandler(log_file, maxBytes=1_000_000, backupCount=3)
            fh.setLevel(level)
            fh.setFormatter(formatter)
            logger.addHandler(fh)




    return logger

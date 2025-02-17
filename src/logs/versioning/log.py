import logging
import subprocess

class Logger:
    def __init__(self, log_file='training.log'):
        self.logger = logging.getLogger('training_logger')
        self.logger.setLevel(logging.INFO)

        log_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(log_format)
        self.logger.addHandler(file_handler)

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(log_format)
        self.logger.addHandler(console_handler)

        self.commit_hash = self.get_git_commit_hash()
        self.log(f"Versão do código: {self.commit_hash}")

    def get_git_commit_hash(self):
        """Obtém o hash do último commit do repositório Git"""
        try:
            commit_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('utf-8').strip()
            return commit_hash
        except Exception as e:
            return f"Erro ao obter commit: {e}"

    def log(self, message):
        """Logar uma mensagem incluindo a versão do código"""
        self.logger.info(f"[{self.commit_hash}] {message}")

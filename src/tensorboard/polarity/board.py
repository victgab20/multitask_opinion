from torch.utils.tensorboard import SummaryWriter

class TensorBoardLogger:
    def __init__(self, log_dir='runs/experiment'):
        # Cria um writer para o TensorBoard
        self.writer = SummaryWriter(log_dir)

    def log_metrics(self, epoch, train_loss, train_accuracy, dev_loss, dev_accuracy, train_precision, train_recall, train_f1, dev_precision, dev_recall, dev_f1):
        # Logar métricas para o TensorBoard
        self.writer.add_scalar('Loss/Train', train_loss, epoch)
        self.writer.add_scalar('Accuracy/Train', train_accuracy, epoch)
        self.writer.add_scalar('Loss/Dev', dev_loss, epoch)
        self.writer.add_scalar('Accuracy/Dev', dev_accuracy, epoch)
        self.writer.add_scalar('Precision/Train', train_precision, epoch)
        self.writer.add_scalar('Recall/Train', train_recall, epoch)
        self.writer.add_scalar('F1/Train', train_f1, epoch)
        self.writer.add_scalar('Precision/Dev', dev_precision, epoch)
        self.writer.add_scalar('Recall/Dev', dev_recall, epoch)
        self.writer.add_scalar('F1/Dev', dev_f1, epoch)

    def close(self):
        self.writer.close()

import torch
from tqdm import tqdm


class Trainer:
    def __init__(self, model, optimizer=None, criterion=None, device=None):
        """Initialize the trainer"""
        self.model = model
        self.optimizer = optimizer or torch.optim.Adam(model.parameters(), lr=0.001)
        self.criterion = criterion or torch.nn.CrossEntropyLoss()
        self.device = device or "cpu"
        self.model = self.model.to(device)

    def get_model(self):
        return self.model

    def train(self, num_epochs, train_dataloader, val_dataloader=None, mlflow=None):
        """Trains the model and logs the results"""
        for epoch in tqdm(range(num_epochs)):
            train_loss, train_acc = self.train_epoch(dataloader=train_dataloader)
            if mlflow:
                mlflow.log_metric("train_loss", train_loss, step=epoch)
                mlflow.log_metric("train_acc", train_acc, step=epoch)
            # Validate only if we have a val dataloader
            if val_dataloader is not None:
                val_loss, val_acc = self.eval_epoch(dataloader=val_dataloader)
                if mlflow:
                    mlflow.log_metric("val_loss", val_loss, step=epoch)
                    mlflow.log_metric("val_acc", val_acc, step=epoch)

    def train_epoch(self, dataloader):
        """Trains one epoch"""
        self.model.train()
        total_loss = 0.0
        total_correct = 0.0
        for i, batch in enumerate(dataloader):
            # Send to device
            X, y = batch
            X = X.to(self.device)
            y = y.to(self.device)

            # Train step
            self.optimizer.zero_grad()  # Clear gradients.
            outs = self.model(X)  # Perform a single forward pass.
            loss = self.criterion(outs, y)

            loss.backward()  # Derive gradients.
            self.optimizer.step()  # Update parameters based on gradients.

            # Compute metrics
            total_loss += loss.detach().item()
            total_correct += torch.sum(torch.argmax(outs, dim=-1) == y).detach().item()
        total_acc = total_correct / (len(dataloader) * dataloader.batch_size)
        return total_loss, total_acc

    def eval_epoch(self, dataloader):
        self.model.eval()
        total_loss = 0.0
        total_correct = 0.0
        for i, batch in enumerate(dataloader):
            # Send to device
            X, y = batch
            X = X.to(self.device)
            y = y.to(self.device)

            # Eval
            outs = self.model(X)
            loss = self.criterion(outs, y)

            # Compute metrics
            total_loss += loss.detach().item()
            total_correct += torch.sum(torch.argmax(outs, dim=-1) == y).detach().item()
        total_acc = total_correct / (len(dataloader) * dataloader.batch_size)
        return total_loss, total_acc

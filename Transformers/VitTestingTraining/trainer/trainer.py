import torch
from utils.utils import save_checkpoint, save_experiment
from torch.utils.tensorboard import SummaryWriter  # For TensorBoard visualization (optional)
import time

class Trainer:
    """
    The simple trainer.
    """

    def __init__(self, model, optimizer, loss_fn, exp_name, device):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.exp_name = exp_name
        self.device = device
        self.writer = SummaryWriter(f'runs/{exp_name}')

    def train(self, trainloader, testloader, epochs, config, save_model_every_n_epochs=0):
        train_losses = []
        test_losses = []
        accuracies = []

        for epoch in range(epochs):
            start_time = time.time()
            # Training loop
            self.model.train()
            running_loss = 0.0
            for i, data in enumerate(trainloader):
                inputs, labels = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                # Zero the parameter gradients
                self.optimizer.zero_grad()

                # Forward pass
                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, labels)

                # Backward pass and optimization
                loss.backward()
                self.optimizer.step()

                # Accumulate loss
                running_loss += loss.item()

                # Print statistics every 100 batches
                if (i + 1) % 100 == 0:
                    avg_loss = running_loss / 100
                    print(f'Epoch [{epoch + 1}/{epochs}], Step [{i + 1}/{len(trainloader)}], Loss: {avg_loss:.4f}')
                    # Optional: Log to TensorBoard
                    self.writer.add_scalar('Training Loss', avg_loss, epoch * len(trainloader) + i)
                    running_loss = 0.0

            # Validation loop
            self.model.eval()
            val_loss = 0.0
            correct = 0
            total = 0
            with torch.no_grad():
                for images, labels in testloader:
                    images, labels = images.to(self.device), labels.to(self.device)
                    outputs = self.model(images)
                    loss = self.loss_fn(outputs, labels)
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            # Calculate average losses and accuracy
            avg_train_loss = running_loss / len(trainloader)
            avg_val_loss = val_loss / len(testloader)
            accuracy = 100 * correct / total

            train_losses.append(avg_train_loss)
            test_losses.append(avg_val_loss)
            accuracies.append(accuracy)

            # Print epoch summary
            epoch_time = time.time() - start_time
            print(f'Epoch [{epoch + 1}/{epochs}] completed in {epoch_time:.2f}s')
            print(f'Training Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}, Accuracy: {accuracy:.2f}%\n')

            # Optional: Log to TensorBoard
            self.writer.add_scalar('Validation Loss', avg_val_loss, epoch)
            self.writer.add_scalar('Accuracy', accuracy, epoch)

            # Save model checkpoint if needed
            if save_model_every_n_epochs > 0 and (epoch + 1) % save_model_every_n_epochs == 0:
                save_checkpoint(self.exp_name, self.model, epoch + 1)

        # After training is complete
        print('Training complete!')
        save_experiment(self.exp_name, config, self.model, train_losses, test_losses, accuracies)
        # Close TensorBoard writer
        self.writer.close()

    def train_epoch(self, trainloader):
        """
        Train the model for one epoch.
        """
        self.model.train()
        total_loss = 0
        for batch in trainloader:
            # Move the batch to the device
            batch = [t.to(self.device) for t in batch]
            images, labels = batch
            # Zero the gradients
            self.optimizer.zero_grad()
            # Calculate the loss
            loss = self.loss_fn(self.model(images)[0], labels)
            # Backpropagate the loss
            loss.backward()
            # Update the model's parameters
            self.optimizer.step()
            total_loss += loss.item() * len(images)
        return total_loss / len(trainloader.dataset)

    @torch.no_grad()
    def evaluate(self, testloader):
        self.model.eval()
        total_loss = 0
        correct = 0
        with torch.no_grad():
            for batch in testloader:
                # Move the batch to the device
                batch = [t.to(self.device) for t in batch]
                images, labels = batch

                # Get predictions
                logits, _ = self.model(images)

                # Calculate the loss
                loss = self.loss_fn(logits, labels)
                total_loss += loss.item() * len(images)

                # Calculate the accuracy
                predictions = torch.argmax(logits, dim=1)
                correct += torch.sum(predictions == labels).item()
        accuracy = correct / len(testloader.dataset)
        avg_loss = total_loss / len(testloader.dataset)
        return accuracy, avg_loss

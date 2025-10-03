import os, sys, time
import numpy as np
import matplotlib.pyplot as plt
import torch
import tensorboard
import uproot
from alive_progress import alive_bar


class MLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(10, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 2)
        )

    def forward(self, x):
        return self.layers(x)

def create_dummy_dataset(total_events=1000000):
    # Dummy data for demonstration
    # Each dummy event (index) should consist of 2x1 matrix inputs with 5 channels in the form of a numpy array. Each truth label shoul;d be a 2x1 matrix with 1 channel.
    # Lets create a dummy dataset: let channel 1,3,5 have small variations from the truth label, and channel 2,4 have large variations.
    # Lets create 7.5m train, 1.25m validation, and 1.25m test events.
    num_train_events = int(0.75 * total_events)
    num_val_events = int(0.125 * total_events)
    num_test_events = int(0.125 * total_events)
    num_channels = 5
    input_shape = (2, 1, num_channels)
    truth_shape = (2, 1, 1)

    # Create random truth values for all events
    np.random.seed(42)  # For reproducibility

    # Generate truth values: 2x1 matrices with 1 channel
    truth_values = np.random.normal(0, 1, (total_events, 2, 1, 1))

    # Create input values that vary from truth with different magnitudes per channel
    input_values = np.zeros((total_events, 2, 1, num_channels))

    # Define variation scales for each channel (channels 1,3,5 small, 2,4 large)
    variations = [0.1, 0.5, 0.1, 0.5, 0.1]  # Index 0-4 for channels 1-5

    # Generate systematic offsets for each event
    systematic_offsets = np.random.normal(0, 0.2, (total_events, 1, 1, 1))

    # Fill input values with variations from truth
    for ch in range(num_channels):
        # Broadcast truth to all channels and add random variations
        channel_noise = np.random.normal(0, variations[ch], (total_events, 2, 1, 1))
        # Add truth + noise + systematic offset
        input_values[:, :, :, ch:ch+1] = truth_values + channel_noise + systematic_offsets

    # Split into train, validation, and test sets
    train_inputs = input_values[:num_train_events]
    train_truths = truth_values[:num_train_events]

    val_inputs = input_values[num_train_events:num_train_events+num_val_events]
    val_truths = truth_values[num_train_events:num_train_events+num_val_events]

    test_inputs = input_values[num_train_events+num_val_events:]
    test_truths = truth_values[num_train_events+num_val_events:]
    
    print(f"Created dataset with {total_events} events:")
    print(f"  Train: {num_train_events} events. Shape: {train_inputs.shape}, {train_truths.shape}")
    print(f"  Validation: {num_val_events} events. Shape: {val_inputs.shape}, {val_truths.shape}")
    print(f"  Test: {num_test_events} events. Shape: {test_inputs.shape}, {test_truths.shape}")

    return (train_inputs, train_truths), (val_inputs, val_truths), (test_inputs, test_truths)


def main():
    # Create output directory if it doesn't exist
    output_dir = f"./model_runs/run_{time.strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(output_dir, exist_ok=True)

    # Hyperparameters
    num_epochs = 100
    batch_size = 1024
    lr = 1e-2

    num_workers = 4  # Number of workers for DataLoader
    
    # Initialisation
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLP().to(device)
    criterion = torch.nn.MSELoss()
    #criterion = torch.nn.L1Loss()  # Using L1 loss for regression tasks
    optimizer = torch.optim.Adam(model.parameters(), lr)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

        # Dataset creation - convert to PyTorch tensors first
    train_dataset, val_dataset, test_dataset = create_dummy_dataset(1000000)
    
    # Convert numpy arrays to PyTorch tensors for better DataLoader efficiency
    train_inputs_tensor = torch.from_numpy(train_dataset[0]).float()
    train_truths_tensor = torch.from_numpy(train_dataset[1]).float()
    val_inputs_tensor = torch.from_numpy(val_dataset[0]).float()
    val_truths_tensor = torch.from_numpy(val_dataset[1]).float()
    
    # Create TensorDataset for proper DataLoader usage
    train_tensor_dataset = torch.utils.data.TensorDataset(train_inputs_tensor, train_truths_tensor)
    val_tensor_dataset = torch.utils.data.TensorDataset(val_inputs_tensor, val_truths_tensor)
    
    train_loader = torch.utils.data.DataLoader(
        train_tensor_dataset,
        batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_tensor_dataset,
        batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )

    # Training loop
    train_loss = []
    val_loss = []

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")

        model.train()
        running_loss = 0.0
        total_iterations = len(train_loader)
        with alive_bar(total_iterations, force_tty=True) as bar:
            for i, (inputs, truths) in enumerate(train_loader):  # Properly unpack batch
                # Move data to device
                inputs = inputs.to(device, non_blocking=True)
                truths = truths.to(device, non_blocking=True)
                
                # Reshape inputs for MLP: flatten the 2x1x5 to 10 features per sample
                inputs = inputs.view(inputs.size(0), -1)  # (batch_size, 10)
                truths = truths.view(truths.size(0), -1)   # (batch_size, 2)
                
                # Zero gradients
                optimizer.zero_grad()
                
                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, truths)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                bar()  # Update progress bar
        
        # Calculate average training loss
        avg_train_loss = running_loss / len(train_loader)
        train_loss.append(avg_train_loss)
        
        # Validation phase
        model.eval()
        val_running_loss = 0.0
        with torch.no_grad():
            for inputs, truths in val_loader:
                inputs = inputs.to(device, non_blocking=True)
                truths = truths.to(device, non_blocking=True)
                
                # Reshape inputs and truths
                inputs = inputs.view(inputs.size(0), -1)
                truths = truths.view(truths.size(0), -1)
                
                outputs = model(inputs)
                val_loss_batch = criterion(outputs, truths)
                val_running_loss += val_loss_batch.item()
        
        avg_val_loss = val_running_loss / len(val_loader)
        val_loss.append(avg_val_loss)
        
        print(f"Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")
        
        # Step the learning rate scheduler
        lr_scheduler.step()

    print("Training complete.")

    # Save the model
    torch.save(model.state_dict(), os.path.join(output_dir, "model.pth"))
    print(f"Model saved to {output_dir}/model.pth")

    # Save train, valid losses in matplotlib format
    plt.figure(figsize=(10, 5))
    plt.plot(train_loss, label='Train Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig(os.path.join(output_dir, "loss_plot.png"))



if __name__ == "__main__":
    # Example usage of the imported libraries
    print("NumPy version:", np.__version__)
    print("PyTorch version:", torch.__version__)
    print("TensorBoard version:", tensorboard.__version__)
    print("Uproot version:", uproot.__version__)
    
    main()

import os
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm

def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class IMUSequence:
    def __init__(self, imu_data, vi_data, name):
        self.imu_data = imu_data
        self.vi_data = vi_data
        self.name = name
        

def load_data(root_dir):
    sequences = []
    for data_folder in os.listdir(root_dir):
        folder_path = os.path.join(root_dir, data_folder) #change: remove 'syn'
        if os.path.isdir(folder_path):
            
            imu = pd.read_csv(os.path.join(folder_path, 'mag.csv')).iloc[1:, 1:].values #change
            vi = pd.read_csv(os.path.join(folder_path, 'gt.csv')).iloc[1:, 1:].values #change
            sequences.append(IMUSequence(imu, vi, f"{data_folder}"))
            
    return sequences

def evaluate_model(model, sequences, sequence_length, output_size):
    device = get_device()
    model.to(device)
    model.eval()
    total_loss = 0
    all_predictions = []
    all_targets = []    
    with torch.no_grad():
        for seq_idx, sequence in enumerate(tqdm(sequences, desc="Evaluating")):
            seq_predictions = []
            seq_targets = sequence.vi_data
            
            for i in range(len(sequence.imu_data)):
                if i < sequence_length - 1:
                    # Pad with zeros for initial sequence
                    pad_length = sequence_length - 1 - i
                    imu_seq = np.pad(sequence.imu_data[:i+1], ((pad_length, 0), (0, 0)), mode='constant')
                    output = torch.zeros(output_size, device=device)
                else:
                    imu_seq = sequence.imu_data[i-sequence_length+1:i+1]
                    imu_seq = torch.FloatTensor(imu_seq).unsqueeze(0).to(device)  # Add batch dimension and move to device
                    output = model(imu_seq).squeeze(0)
                
                seq_predictions.append(output.cpu().numpy())
                
                loss = torch.nn.functional.mse_loss(output, torch.FloatTensor(seq_targets[i]).to(device))
                total_loss += loss.item()
            
            seq_predictions = np.array(seq_predictions)
            seq_targets = np.array(seq_targets)
            
            # Calculate MAE and MSE for the sequence
            seq_mae = np.mean(np.abs(seq_predictions - seq_targets))
            seq_mse = np.mean((seq_predictions - seq_targets) ** 2)
            
            print(f"\nSequence: {sequence.name}")
            print(f"MAE: {seq_mae:.4f}")
            print(f"MSE: {seq_mse:.4f}")
            
            all_predictions.extend(seq_predictions)
            all_targets.extend(seq_targets)
    
    avg_loss = total_loss / sum(len(seq.imu_data) for seq in sequences)
    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)
    
    return avg_loss, all_predictions, all_targets

def test_model(model, test_root_dir, sequence_length, output_size):
    print("\nStarting model evaluation...")

    # Load and evaluate test data
    sequences = load_data(test_root_dir)

    print("Testing Dataset Information:")
    print(f"Number of sequences: {len(sequences)}")
    print(f"Model Sequence length: {sequence_length}")

    test_loss, predictions, targets = evaluate_model(model, sequences, sequence_length, output_size)
    print(f"\nOverall Test Loss: {test_loss:.4f}")
    print(f"Predictions shape: {predictions.shape}")
    print(f"Targets shape: {targets.shape}")

    # Calculate and print overall metrics
    overall_mse = np.mean((predictions - targets) ** 2)
    overall_mae = np.mean(np.abs(predictions - targets))
    print(f"\nOverall Mean Squared Error: {overall_mse:.4f}")
    print(f"Overall Mean Absolute Error: {overall_mae:.4f}")

    return test_loss, overall_mse, overall_mae


# test_root_dir = 'processed/test/path2'
# sequences = load_data(test_root_dir)
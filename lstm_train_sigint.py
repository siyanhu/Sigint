import argparse
import torch
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
from src.lstm_model import IMULSTMModel #change
from tcn_model import IMUTCNModel #change
from prepare_seq import prepare_data #change
from lstm_test_sigint import test_model #change
from torch.utils.tensorboard import SummaryWriter
import datetime
import os
from torch.optim.lr_scheduler import ReduceLROnPlateau

def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def compute_gradient_norm(model):
    total_norm = 0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    return total_norm

def load_model(model_type, **model_params):  # New function
    if model_type.lower() == 'lstm':
        return IMULSTMModel(**model_params)
    elif model_type.lower() == 'tcn':
        return IMUTCNModel(**model_params)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    
def train_and_evaluate(args):
    # Create a custom log directory name
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    #log_dir = os.path.join("./lstm_logs", f"{current_time}_input{args.input_size}_hidden{'-'.join(map(str, args.hidden_sizes))}_output{args.output_size}_lr{args.learning_rate}_batch{args.batch_size}_dropout{args.dropout_rate}_sequencelength{args.sequence_length}_transform{args.transform}")    
    log_dir = os.path.join(f"./{args.model_type}_logs", f"{current_time}_input{args.input_size}_hidden{'-'.join(map(str, args.hidden_sizes))}_output{args.output_size}_lr{args.learning_rate}_batch{args.batch_size}_dropout{args.dropout_rate}_sequencelength{args.sequence_length}_transform{args.transform}")  
    writer = SummaryWriter(log_dir)

    # Load data
    train_loader, val_loader, train_dataset, val_dataset = prepare_data(args.root_dir, args.sequence_length, args.batch_size, args.stride)

    # Print and log data information
    print(f"Total training samples: {len(train_dataset)}")
    print(f"Total validation samples: {len(val_dataset)}")
    print(f"Total samples: {len(train_dataset) + len(val_dataset)}")
    print(f"Input shape: {train_dataset[0][0].shape}")
    print(f"Target shape: {train_dataset[0][1].shape}")
    print(f"Number of training batches: {len(train_loader)}")
    print(f"Number of validation batches: {len(val_loader)}")
    print(f"Transform: {args.transform}")
    print(f"Sride: {args.stride}")

    writer.add_text("Dataset Info", f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    writer.add_text("Input Shape", str(train_dataset[0][0].shape))
    writer.add_text("Target Shape", str(train_dataset[0][1].shape))
    writer.add_text("Transform", str(args.transform))
    writer.add_text("Sride", str(args.stride))


    # Initialize the model, loss function, and optimizer
    device = get_device()
    print(f"Using device: {device}")
    
    model = load_model(args.model_type, input_size=args.input_size, hidden_sizes=args.hidden_sizes, 
                       output_size=args.output_size, dropout_rate=args.dropout_rate).to(device)  # Changed 
       
    #criterion = torch.nn.MSELoss()
    criterion = torch.nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, min_lr=1e-6) #change

    def mean_baseline_eval(loader, mean_position):
        total_loss = 0
        for _, targets in loader:
            batch_size = targets.size(0)
            mean_predictions = mean_position.repeat(batch_size, 1)
            loss = torch.nn.functional.mse_loss(mean_predictions, targets)
            total_loss += loss.item() * batch_size
        return total_loss / len(loader.dataset)

    # Calculate mean position from the training set
    all_targets = torch.cat([targets for _, targets in train_loader], dim=0)
    mean_position = all_targets.mean(dim=0)

    # Perform mean baseline evaluation
    print("Performing mean baseline evaluation...")
    baseline_train_loss = mean_baseline_eval(train_loader, mean_position)
    baseline_val_loss = mean_baseline_eval(val_loader, mean_position)
    print(f"Baseline Train Loss: {baseline_train_loss:.4f}, Baseline Val Loss: {baseline_val_loss:.4f}")

    # Log baseline loss as "Epoch 0"
    writer.add_scalar("Loss/Train", baseline_train_loss, 0)
    writer.add_scalar("Loss/Validation", baseline_val_loss, 0)
    

    # Training loop
    best_val_loss = float('inf')
    early_stopping_counter = 0 #change
    early_stopping_patience = 20
    
    for epoch in range(args.num_epochs):
        model.train()
        train_loss = 0
        #epoch_grad_norm = 0 #change
        for batch_idx, (imu_seq, vi_target) in enumerate(train_loader):
            imu_seq, vi_target = imu_seq.to(device), vi_target.to(device)
            outputs = model(imu_seq)
            loss = criterion(outputs, vi_target)
            
            optimizer.zero_grad()
            loss.backward()
            
            # Compute gradient norm
            grad_norm = compute_gradient_norm(model)
            #epoch_grad_norm += grad_norm
            
            optimizer.step()
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0
        val_mae = 0
        with torch.no_grad():
            for imu_seq, vi_target in val_loader:
                imu_seq, vi_target = imu_seq.to(device), vi_target.to(device)
                outputs = model(imu_seq)
                val_loss += criterion(outputs, vi_target).item()
                
                val_mae += torch.mean(torch.abs(outputs - vi_target)).item()
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        val_mae /= len(val_loader) #change
        #avg_grad_norm = epoch_grad_norm / len(train_loader) #change
        
        # Print average gradient norm for the epoch
        print(f'Epoch [{epoch+1}/{args.num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val MAE: {val_mae:.4f}')
       
        
        # Log to TensorBoard
        writer.add_scalar("Loss/Train", train_loss, epoch + 1)
        writer.add_scalar("Loss/Validation", val_loss, epoch + 1)
        writer.add_scalar("Validation/MAE", val_mae, epoch + 1)
        #writer.add_scalar("Gradient Norm", avg_grad_norm, epoch + 1)
        
        
        # Update the scheduler
        scheduler.step(val_loss)
        
        # Check if learning rate has changed
        current_lr = optimizer.param_groups[0]['lr']
        if current_lr != args.learning_rate:
            print(f'Epoch {epoch + 1}: Learning rate set to {current_lr}')
            args.learning_rate = current_lr
            
        writer.add_scalar("Learning Rate", current_lr, epoch + 1)
        
        # Save the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.cpu().state_dict(), args.model_save_path)
            model.to(device)  # Move the model back to the device
            writer.add_scalar("Best_Val_Loss", best_val_loss, epoch + 1)
        else:
            early_stopping_counter += 1
        
        if early_stopping_counter >= early_stopping_patience:
            print(f"Early stopping triggered after {epoch + 1} epochs")
            break

    print("Training completed.")

    # Testing
    print("\nStarting model evaluation...")
    
    # Load the best model
    model.load_state_dict(torch.load(args.model_save_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()

    test_loss, overall_mse, overall_mae = test_model(model, args.test_root_dir, args.sequence_length, args.output_size)

    # Log test results to TensorBoard
    writer.add_scalar("Test/Loss", test_loss, 0)
    writer.add_scalar("Test/MSE", overall_mse, 0)
    writer.add_scalar("Test/MAE", overall_mae, 0)

    # Add text summary of test results
    writer.add_text("Test Results", 
                    f"Test Loss: {test_loss:.4f}\n"
                    f"Overall MSE: {overall_mse:.4f}\n"
                    f"Overall MAE: {overall_mae:.4f}")

    print(f"Test results logged to TensorBoard in {log_dir}")
    
    #save the model 
    model_path = f'./model/{args.model_type}_model_length_{args.sequence_length}_transform_{args.transform}_{overall_mae:.4f}.pth'
    torch.save(model.cpu().state_dict(), model_path)
    print(f"Save model to {model_path}")

    writer.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate IMU model for IMU data")
    parser.add_argument("--model_type", type=str, required=True, choices=['lstm', 'tcn'], help="Type of model to train")  # New argument
    parser.add_argument("--root_dir", type=str, default="./processed/no_transform/new/train", help="Root directory of the training dataset") #change
    parser.add_argument("--sequence_length", type=int, default=100, help="Sequence length for model input")
    parser.add_argument("--input_size", type=int, default=3, help="Number of features in IMU data") #change
    parser.add_argument("--hidden_sizes", type=int, nargs='+', default=[64, 32], help="Hidden sizes of LSTM layers") 
    parser.add_argument("--output_size", type=int, default=2, help="Output size (x, y)") #change
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--num_epochs", type=int, required=True, help="Number of epochs")
    parser.add_argument("--dropout_rate", type=float, default=0.2, help="Dropout rate")
    parser.add_argument("--model_save_path", type=str, default="./model/best_imu_model.pth", help="Path to save the best model")
    parser.add_argument("--test_root_dir", type=str, default="./processed/no_transform/new/test", help="Root directory of the test dataset") #change
    parser.add_argument("--transform", type=bool, default=False, help="Transform or not") #change
    parser.add_argument("--stride", type=int, default=10, help="Stride") #change
   
    args = parser.parse_args()
    train_and_evaluate(args)
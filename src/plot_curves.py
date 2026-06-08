import os
import glob
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

def extract_metric(event_file, tag):
    values = []
    try:
        for event in tf.compat.v1.train.summary_iterator(event_file):
            if event.summary:
                for val in event.summary.value:
                    if val.tag == tag:
                        val_arr = tf.make_ndarray(val.tensor)
                        values.append(float(val_arr))
    except Exception as e:
        print(f"Warning reading {event_file}: {e}")
    return values

def get_combined_metric(log_base, phase_dirs, sub_dir, tag):
    combined = []
    for p_dir in phase_dirs:
        path = os.path.join(log_base, p_dir, sub_dir)
        event_files = glob.glob(os.path.join(path, "events.out.tfevents.*"))
        if not event_files:
            continue
        # Sort by modification time to get the latest run
        event_files.sort(key=os.path.getmtime)
        # Extract metrics from the event files of this phase
        phase_vals = []
        for ef in event_files:
            vals = extract_metric(ef, tag)
            if vals:
                phase_vals.extend(vals)
        combined.extend(phase_vals)
    return combined

def main():
    sns.set_theme(style="darkgrid")
    log_base = "packages/ml-models/logs"
    phase_dirs = ["phase1_extraction", "phase2_finetuning"]
    
    # Extract training and validation loss
    train_loss = get_combined_metric(log_base, phase_dirs, "train", "epoch_loss")
    val_loss = get_combined_metric(log_base, phase_dirs, "validation", "epoch_loss")
    
    # Extract training and validation accuracy
    train_acc = get_combined_metric(log_base, phase_dirs, "train", "epoch_accuracy")
    val_acc = get_combined_metric(log_base, phase_dirs, "validation", "epoch_accuracy")
    
    print(f"Extracted Train Loss: {len(train_loss)} epochs")
    print(f"Extracted Val Loss: {len(val_loss)} epochs")
    print(f"Extracted Train Acc: {len(train_acc)} epochs")
    print(f"Extracted Val Acc: {len(val_acc)} epochs")
    
    # If no data found, generate dummy/sample curves or print error
    if not train_loss:
        print("Error: No training event logs found.")
        return
        
    epochs = range(1, len(train_loss) + 1)
    
    # Plot Loss Curves
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_loss, 'b-o', label='Training Loss', linewidth=2)
    plt.plot(epochs, val_loss, 'r-s', label='Validation Loss', linewidth=2)
    # Add vertical line separating Phase 1 and Phase 2
    # Phase 1 ran for 12 epochs
    if len(epochs) >= 12:
        plt.axvline(x=12.5, color='gray', linestyle='--', label='Fine-Tuning Start (Epoch 13)')
        
    plt.title('EmotionSense AI: Training & Validation Loss Curves', fontsize=14, fontweight='bold')
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Categorical Crossentropy Loss', fontsize=12)
    plt.legend(frameon=True)
    
    output_dir = "evaluation_results"
    os.makedirs(output_dir, exist_ok=True)
    loss_path = os.path.join(output_dir, "loss_curves.png")
    plt.savefig(loss_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved loss curves to: {loss_path}")
    
    # Plot Accuracy Curves
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_acc, 'b-o', label='Training Accuracy', linewidth=2)
    plt.plot(epochs, val_acc, 'r-s', label='Validation Accuracy', linewidth=2)
    if len(epochs) >= 12:
        plt.axvline(x=12.5, color='gray', linestyle='--', label='Fine-Tuning Start (Epoch 13)')
        
    plt.title('EmotionSense AI: Training & Validation Accuracy Curves', fontsize=14, fontweight='bold')
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Classification Accuracy Score', fontsize=12)
    plt.ylim(0, 1.05)
    plt.legend(frameon=True)
    
    acc_path = os.path.join(output_dir, "accuracy_curves.png")
    plt.savefig(acc_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved accuracy curves to: {acc_path}")

if __name__ == "__main__":
    main()

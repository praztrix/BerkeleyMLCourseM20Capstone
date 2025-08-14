# utils.py

import os
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report


def mount_google_drive():
    """
    Conditionally mounts Google Drive if the code is running in a Colab environment.
    """
    try:
        from google.colab import drive
        print("Mounting Google Drive...")
        drive.mount('/content/drive')
    except ImportError:
        print("This environment is not Google Colab. Skipping Google Drive mounting.")
        pass


def get_file_or_dir_size(path):
    """Calculates the total size of a file or directory."""
    if os.path.isfile(path):
        return os.path.getsize(path)
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            total_size += os.path.getsize(fp)
    return total_size


def get_in_memory_model_size(model):
    """
    Calculates the size of a Keras model in memory by summing the bytes of its weights.

    Args:
        model (tf.keras.Model): The model to calculate the size for.

    Returns:
        float: The size of the model in megabytes (MB).
    """
    print("Calculating model size in memory...")
    total_size = 0
    
    for layer in model.layers:
        weights = layer.get_weights()
        for weight in weights:
            if isinstance(weight, np.ndarray):
                total_size += weight.nbytes

    return total_size / (1024 * 1024)


def recreate_plots_from_results(results_dict, CLASS_NAMES): 
    """
    Recreates plots from the plotting data stored in the results dictionary.

    Args:
        results_dict (dict): The dictionary returned by the run_model_pipeline function.
        CLASS_NAMES (list): List of class names for plotting.
    """
    model_name = results_dict['model_name']
    plotting_data = results_dict['plotting_data']

    print(f"\nRecreating plots for {model_name} from stored results...")

    # --- Plot Loss and Accuracy Curves ---
    if plotting_data['history']:
        print("  - Recreating Loss and Accuracy Curves.")
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.plot(plotting_data['history']['loss'], label='Training Loss')
        plt.plot(plotting_data['history']['val_loss'], label='Validation Loss')
        plt.title(f'Loss Curves for {model_name}')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(plotting_data['history']['accuracy'], label='Training Accuracy')
        plt.plot(plotting_data['history']['val_accuracy'], label='Validation Accuracy')
        plt.title(f'Accuracy Curves for {model_name}')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.tight_layout()
        plt.show()
    else:
        print("  - Skipping Loss/Accuracy plots: No history data found.")

    # --- Plot Confusion Matrix ---
    if plotting_data['y_true_classes'] is not None and plotting_data['y_pred_classes'] is not None:
        print("  - Recreating Confusion Matrix.")
        cm = confusion_matrix(
            y_true=plotting_data['y_true_classes'],
            y_pred=plotting_data['y_pred_classes'],
            labels=range(len(CLASS_NAMES)) 
        )

        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=CLASS_NAMES, 
                    yticklabels=CLASS_NAMES) 
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title(f'Confusion Matrix for {model_name} on CIFAR-10 Test Data (Recreated)')
        plt.show()
    else:
        print("  - Skipping Confusion Matrix plot: Missing true or predicted class labels.")




import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def plot_all_model_metrics(all_results):
    """
    Plots a comparison of key metrics for all models in a structured way.

    Args:
        all_results (dict): A dictionary containing results from all model pipelines.
    """
    print("\n\n====================== Plotting All Model Metrics for Comparison ======================")

    
    ordered_metric_keys = [
        'Total Parameters',
        'Training Time (s)',
        'Tuner Search Time (s)',
        'Train Accuracy',
        'Train Loss',
        'Validation Accuracy',
        'Validation Loss',
        'CINIC-10 Test Accuracy',  
        'CINIC-10 Test Loss',      
        'Inference Time Per Sample (s)',
        'In-Memory Model Size (MB)',
        'File-Based Model Size (MB)'
    ]

    # Extract metrics into a convenient format for plotting
    metrics_to_plot = {key: {} for key in ordered_metric_keys}

    for model_name, result in all_results.items():
        # Only extract the metrics we intend to plot
        metrics_to_plot['Training Time (s)'][model_name] = result['metrics'].get('training_time')
        metrics_to_plot['Total Parameters'][model_name] = result['metrics'].get('total_parameters')
        metrics_to_plot['Tuner Search Time (s)'][model_name] = result['metrics'].get('tuner_search_time')
        metrics_to_plot['Train Accuracy'][model_name] = result['metrics'].get('cifar10_train_accuracy')
        metrics_to_plot['Train Loss'][model_name] = result['metrics'].get('cifar10_train_loss')
        metrics_to_plot['Validation Accuracy'][model_name] = result['metrics'].get('cifar10_val_accuracy')
        metrics_to_plot['Validation Loss'][model_name] = result['metrics'].get('cifar10_val_loss')
        metrics_to_plot['CINIC-10 Test Accuracy'][model_name] = result['metrics'].get('cinic10_test_accuracy') 
        metrics_to_plot['CINIC-10 Test Loss'][model_name] = result['metrics'].get('cinic10_test_loss')         
        metrics_to_plot['Inference Time Per Sample (s)'][model_name] = result['metrics'].get('inference_time_per_sample')
        metrics_to_plot['In-Memory Model Size (MB)'][model_name] = result['metrics'].get('in_memory_model_size_mb')
        metrics_to_plot['File-Based Model Size (MB)'][model_name] = result['metrics'].get('file_based_model_size_mb')

    df_metrics = pd.DataFrame(metrics_to_plot)

    sns.set_style("whitegrid")

    # Create a figure with a subplot for each metric
    fig, axes = plt.subplots(nrows=4, ncols=3, figsize=(20, 24))
    fig.suptitle('Model Performance Comparison', fontsize=28, y=1.02)

    # Flatten the axes array for easier iteration
    axes = axes.flatten()


    plot_info = [
        {'col': 'Total Parameters', 'title': 'Total Model Parameters', 'ylabel': 'Parameter Count', 'format': '.1e', 'palette': 'cividis'},
        {'col': 'Training Time (s)', 'title': 'Total Training Time', 'ylabel': 'Time (seconds)', 'format': '.2f', 'palette': 'magma'},
        {'col': 'Tuner Search Time (s)', 'title': 'Hyperparameter Tuner Search Time', 'ylabel': 'Time (seconds)', 'format': '.2f', 'palette': 'plasma'},
        {'col': 'Train Accuracy', 'title': 'Training Accuracy', 'ylabel': 'Accuracy', 'ylim': (0, 1.0), 'format': '.2f', 'palette': 'cividis'},
        {'col': 'Train Loss', 'title': 'Training Loss', 'ylabel': 'Loss', 'format': '.2f', 'palette': 'inferno'},
        {'col': 'Validation Accuracy', 'title': 'Validation Accuracy', 'ylabel': 'Accuracy', 'ylim': (0, 1.0), 'format': '.2f', 'palette': 'viridis'},
        {'col': 'Validation Loss', 'title': 'Validation Loss', 'ylabel': 'Loss', 'format': '.2f', 'palette': 'magma'},
        {'col': 'CINIC-10 Test Accuracy', 'title': 'CINIC-10 Test Accuracy', 'ylabel': 'Accuracy', 'ylim': (0, 1.0), 'format': '.2f', 'palette': 'crest'}, 
        {'col': 'CINIC-10 Test Loss', 'title': 'CINIC-10 Test Loss', 'ylabel': 'Loss', 'format': '.2f', 'palette': 'rocket'}, 
        {'col': 'Inference Time Per Sample (s)', 'title': 'Inference Time Per Sample', 'ylabel': 'Time (seconds)', 'format': '.4f', 'palette': 'mako'},
        {'col': 'In-Memory Model Size (MB)', 'title': 'In-Memory Model Size', 'ylabel': 'Size (MB)', 'format': '.2f', 'palette': 'viridis_r'},
        {'col': 'File-Based Model Size (MB)', 'title': 'File-Based Model Size', 'ylabel': 'Size (MB)', 'format': '.2f', 'palette': 'flare'}
    ]

    for i, p_info in enumerate(plot_info):
        ax = axes[i]
        
        sns.barplot(x=df_metrics.index, y=p_info['col'], data=df_metrics, hue = df_metrics.index,  ax=ax, palette=p_info['palette'], legend=False)
        ax.set_title(p_info['title'], fontsize=14) 
        ax.set_xlabel('Model', fontsize=12)
        ax.set_ylabel(p_info['ylabel'], fontsize=12)
        if 'ylim' in p_info:
            ax.set_ylim(p_info['ylim'])

        for p in ax.patches:
            height = p.get_height()
            ax.annotate(f"{height:{p_info['format']}}",
                        (p.get_x() + p.get_width() / 2., height),
                        ha='center', va='center', xytext=(0, 5), textcoords='offset points', fontsize=10) 
        ax.tick_params(axis='x', rotation=45, labelsize=10) 
        ax.tick_params(axis='y', labelsize=10) # 

    # Hide any unused subplots if the number of plots is less than nrows * ncols
    for j in range(len(plot_info), len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout(rect=[0, 0.03, 1, 0.98]) # Adjust layout to prevent title overlap
    plt.show()




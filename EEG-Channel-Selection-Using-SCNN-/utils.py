"""
Utility functions for EEG data analysis, visualization, and model evaluation
"""
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import logging
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd

logger = logging.getLogger(__name__)


def plot_channel_selection_results(
    selection_results: Dict,
    save_path: Optional[str] = None,
    top_k: int = 10
):
    """
    Visualize channel selection results.
    
    Args:
        selection_results: Results from channel selection
        save_path: Path to save the plot
        top_k: Number of top channels to highlight
    """
    plt.figure(figsize=(15, 10))
    
    # Extract data
    all_results = selection_results['all_results']
    channel_names = [r['channel_name'] for r in all_results]
    accuracies = [r['best_accuracy'] for r in all_results]
    selected_indices = set(selection_results['selected_indices'])
    
    # Create colors (selected vs non-selected)
    colors = ['red' if i in selected_indices else 'lightblue' 
              for i, r in enumerate(all_results)]
    
    # Sort by accuracy for better visualization
    sorted_data = sorted(zip(channel_names, accuracies, colors), 
                        key=lambda x: x[1], reverse=True)
    
    names_sorted, acc_sorted, colors_sorted = zip(*sorted_data)
    
    # Plot 1: Bar chart of all channels
    plt.subplot(2, 2, 1)
    bars = plt.bar(range(len(names_sorted)), acc_sorted, color=colors_sorted)
    plt.title(f'Channel Selection Results (Top {selection_results["selection_summary"]["selected_channels"]} Selected)')
    plt.xlabel('Channel Index (sorted by accuracy)')
    plt.ylabel('SCNN Accuracy')
    plt.xticks(range(0, len(names_sorted), max(1, len(names_sorted)//10)), 
               rotation=45)
    
    # Add legend
    red_patch = plt.Rectangle((0, 0), 1, 1, fc="red", label="Selected")
    blue_patch = plt.Rectangle((0, 0), 1, 1, fc="lightblue", label="Not Selected")
    plt.legend(handles=[red_patch, blue_patch])
    
    # Plot 2: Top K channels detailed view
    plt.subplot(2, 2, 2)
    top_k = min(top_k, len(names_sorted))
    top_names = names_sorted[:top_k]
    top_accs = acc_sorted[:top_k]
    top_colors = colors_sorted[:top_k]
    
    bars = plt.bar(range(top_k), top_accs, color=top_colors)
    plt.title(f'Top {top_k} Channels by SCNN Accuracy')
    plt.xlabel('Channel')
    plt.ylabel('SCNN Accuracy')
    plt.xticks(range(top_k), top_names, rotation=45)
    
    # Add accuracy values on bars
    for i, (bar, acc) in enumerate(zip(bars, top_accs)):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{acc:.3f}', ha='center', va='bottom')
    
    # Plot 3: Accuracy distribution
    plt.subplot(2, 2, 3)
    plt.hist(accuracies, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    plt.axvline(selection_results['selection_summary']['mean_selected_accuracy'], 
                color='red', linestyle='--', linewidth=2, 
                label=f"Mean Selected: {selection_results['selection_summary']['mean_selected_accuracy']:.3f}")
    plt.title('Distribution of Channel Accuracies')
    plt.xlabel('SCNN Accuracy')
    plt.ylabel('Number of Channels')
    plt.legend()
    
    # Plot 4: Selected vs Non-selected comparison
    plt.subplot(2, 2, 4)
    selected_accs = [r['best_accuracy'] for r in all_results 
                    if r['channel_index'] in selected_indices]
    non_selected_accs = [r['best_accuracy'] for r in all_results 
                        if r['channel_index'] not in selected_indices]
    
    data_for_box = [selected_accs, non_selected_accs]
    labels = [f'Selected\n(n={len(selected_accs)})', f'Not Selected\n(n={len(non_selected_accs)})']
    
    box_plot = plt.boxplot(data_for_box, labels=labels, patch_artist=True)
    box_plot['boxes'][0].set_facecolor('red')
    box_plot['boxes'][1].set_facecolor('lightblue')
    
    plt.title('Accuracy Distribution: Selected vs Non-Selected')
    plt.ylabel('SCNN Accuracy')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Channel selection plot saved to {save_path}")
    
    plt.show()


def plot_training_history(
    cv_results: List[Dict],
    save_path: Optional[str] = None
):
    """
    Plot training history for cross-validation folds.
    
    Args:
        cv_results: Cross-validation results from pipeline
        save_path: Path to save the plot
    """
    n_folds = len(cv_results)
    
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Training and validation loss for all folds
    plt.subplot(2, 2, 1)
    for i, fold_result in enumerate(cv_results):
        history = fold_result['train_history']
        epochs = range(1, len(history['train_losses']) + 1)
        plt.plot(epochs, history['train_losses'], '--', alpha=0.7, 
                label=f'Fold {i+1} Train' if i < 3 else None)
        plt.plot(epochs, history['val_losses'], '-', alpha=0.8, 
                label=f'Fold {i+1} Val' if i < 3 else None)
    
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Validation accuracy for all folds
    plt.subplot(2, 2, 2)
    for i, fold_result in enumerate(cv_results):
        history = fold_result['train_history']
        epochs = range(1, len(history['val_accuracies']) + 1)
        plt.plot(epochs, history['val_accuracies'], '-', alpha=0.8,
                label=f'Fold {i+1}' if i < 5 else None)
    
    plt.title('Validation Accuracy Across Folds')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Final accuracy comparison
    plt.subplot(2, 2, 3)
    fold_numbers = [r['fold'] for r in cv_results]
    final_accuracies = [r['accuracy'] for r in cv_results]
    
    bars = plt.bar(fold_numbers, final_accuracies, color='skyblue', edgecolor='black')
    plt.title('Final Accuracy by Fold')
    plt.xlabel('Fold')
    plt.ylabel('Final Accuracy')
    plt.ylim(0, 1)
    
    # Add accuracy values on bars
    for bar, acc in zip(bars, final_accuracies):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{acc:.3f}', ha='center', va='bottom')
    
    # Add mean line
    mean_acc = np.mean(final_accuracies)
    plt.axhline(mean_acc, color='red', linestyle='--', linewidth=2,
                label=f'Mean: {mean_acc:.3f}')
    plt.legend()
    
    # Plot 4: Convergence analysis
    plt.subplot(2, 2, 4)
    convergence_epochs = []
    for fold_result in cv_results:
        history = fold_result['train_history']
        # Find epoch with best validation accuracy
        best_epoch = np.argmax(history['val_accuracies']) + 1
        convergence_epochs.append(best_epoch)
    
    plt.bar(fold_numbers, convergence_epochs, color='lightgreen', edgecolor='black')
    plt.title('Convergence Speed (Epochs to Best Accuracy)')
    plt.xlabel('Fold')
    plt.ylabel('Epochs')
    
    # Add values on bars
    for i, (bar, epochs) in enumerate(zip(plt.gca().patches, convergence_epochs)):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                str(epochs), ha='center', va='bottom')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Training history plot saved to {save_path}")
    
    plt.show()


def plot_frequency_band_analysis(
    fusion_inputs: List[np.ndarray],
    band_names: List[str] = None,
    save_path: Optional[str] = None
):
    """
    Analyze and visualize frequency band characteristics.
    
    Args:
        fusion_inputs: List of filtered inputs for different bands
        band_names: Names of frequency bands
        save_path: Path to save the plot
    """
    if band_names is None:
        band_names = [f'Band {i+1}' for i in range(len(fusion_inputs))]
    
    plt.figure(figsize=(15, 10))
    
    # Calculate statistics for each band
    band_stats = []
    for i, band_data in enumerate(fusion_inputs):
        stats = {
            'name': band_names[i],
            'mean': np.mean(band_data),
            'std': np.std(band_data),
            'min': np.min(band_data),
            'max': np.max(band_data),
            'power': np.mean(band_data**2)
        }
        band_stats.append(stats)
    
    # Plot 1: Mean amplitude by band
    plt.subplot(2, 2, 1)
    means = [stats['mean'] for stats in band_stats]
    stds = [stats['std'] for stats in band_stats]
    
    bars = plt.bar(band_names, means, yerr=stds, capsize=5, 
                  color=['red', 'blue', 'green', 'orange'][:len(band_names)])
    plt.title('Mean Amplitude by Frequency Band')
    plt.ylabel('Amplitude')
    plt.xticks(rotation=45)
    
    # Plot 2: Power by band  
    plt.subplot(2, 2, 2)
    powers = [stats['power'] for stats in band_stats]
    bars = plt.bar(band_names, powers, 
                  color=['red', 'blue', 'green', 'orange'][:len(band_names)])
    plt.title('Average Power by Frequency Band')
    plt.ylabel('Power')
    plt.xticks(rotation=45)
    
    # Plot 3: Distribution comparison
    plt.subplot(2, 2, 3)
    for i, (band_data, name) in enumerate(zip(fusion_inputs, band_names)):
        # Sample data for histogram (to avoid memory issues)
        sample_data = band_data.flatten()[::100]  # Sample every 100th point
        plt.hist(sample_data, bins=50, alpha=0.6, label=name, density=True)
    
    plt.title('Amplitude Distribution by Band')
    plt.xlabel('Amplitude')
    plt.ylabel('Density')
    plt.legend()
    
    # Plot 4: Band characteristics summary
    plt.subplot(2, 2, 4)
    
    # Create a summary table
    summary_data = []
    for stats in band_stats:
        summary_data.append([
            stats['name'],
            f"{stats['mean']:.3f}",
            f"{stats['std']:.3f}",
            f"{stats['power']:.3f}"
        ])
    
    table_data = pd.DataFrame(summary_data, 
                             columns=['Band', 'Mean', 'Std', 'Power'])
    
    # Hide axes and create table
    plt.axis('off')
    table = plt.table(cellText=table_data.values,
                     colLabels=table_data.columns,
                     cellLoc='center',
                     loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    plt.title('Frequency Band Statistics', pad=20)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Frequency band analysis plot saved to {save_path}")
    
    plt.show()


def evaluate_model_predictions(
    model: torch.nn.Module,
    test_loader,
    class_names: List[str] = None,
    device: str = None
) -> Dict:
    """
    Evaluate model predictions and generate classification metrics.
    
    Args:
        model: Trained PyTorch model
        test_loader: Test data loader
        class_names: Names of classes
        device: Computing device
        
    Returns:
        Dictionary with evaluation results
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model.eval()
    all_predictions = []
    all_targets = []
    all_probabilities = []
    
    with torch.no_grad():
        for batch_data in test_loader:
            if len(batch_data) == 2:
                inputs, targets = batch_data
            else:
                inputs = batch_data[:-1]
                targets = batch_data[-1]
            
            # Handle different input formats
            if isinstance(inputs, list):
                inputs = [inp.to(device) for inp in inputs]
                outputs = model(inputs)
            else:
                inputs = inputs.to(device)
                outputs = model(inputs)
            
            targets = targets.to(device)
            
            probabilities = torch.softmax(outputs, dim=1)
            predictions = torch.argmax(outputs, dim=1)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
    
    # Calculate metrics
    accuracy = np.mean(np.array(all_predictions) == np.array(all_targets))
    
    # Classification report
    if class_names is None:
        class_names = [f'Class_{i}' for i in range(len(np.unique(all_targets)))]
    
    report = classification_report(all_targets, all_predictions, 
                                 target_names=class_names, output_dict=True)
    
    # Confusion matrix
    cm = confusion_matrix(all_targets, all_predictions)
    
    results = {
        'accuracy': accuracy,
        'predictions': all_predictions,
        'targets': all_targets,
        'probabilities': np.array(all_probabilities),
        'classification_report': report,
        'confusion_matrix': cm,
        'class_names': class_names
    }
    
    return results


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    save_path: Optional[str] = None,
    normalize: bool = True
):
    """
    Plot confusion matrix.
    
    Args:
        cm: Confusion matrix
        class_names: Class names
        save_path: Path to save plot
        normalize: Whether to normalize the matrix
    """
    plt.figure(figsize=(8, 6))
    
    if normalize:
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        sns.heatmap(cm_norm, annot=True, fmt='.3f', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('Normalized Confusion Matrix')
    else:
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix')
    
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Confusion matrix saved to {save_path}")
    
    plt.show()


def generate_comprehensive_report(
    pipeline_results: Dict,
    output_dir: str = "eeg_analysis_report"
):
    """
    Generate a comprehensive analysis report with all plots and statistics.
    
    Args:
        pipeline_results: Results from run_full_pipeline
        output_dir: Directory to save report files
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info(f"Generating comprehensive analysis report in {output_dir}/")
    
    # Generate all plots
    try:
        # Channel selection visualization
        plot_channel_selection_results(
            pipeline_results['stage1_results']['selection_results'],
            save_path=os.path.join(output_dir, "channel_selection_analysis.png")
        )
        
        # Training history
        plot_training_history(
            pipeline_results['stage2_results']['cv_results'],
            save_path=os.path.join(output_dir, "training_history.png")
        )
        
        logger.info("Analysis report generated successfully!")
        
    except Exception as e:
        logger.error(f"Error generating report: {str(e)}")


def create_summary_statistics_table(pipeline_results: Dict) -> pd.DataFrame:
    """
    Create a summary table of pipeline statistics.
    
    Args:
        pipeline_results: Results from pipeline execution
        
    Returns:
        DataFrame with summary statistics
    """
    # Extract key metrics
    stage1 = pipeline_results['stage1_results']['selection_results']['selection_summary']
    final = pipeline_results['final_results']
    
    summary_data = {
        'Metric': [
            'Total Channels',
            'Selected Channels', 
            'Best SCNN Accuracy',
            'Mean SCNN Accuracy (Selected)',
            'Final Mean CV Accuracy',
            'Final Std CV Accuracy',
            'Best Fold Accuracy',
            'Execution Time (seconds)'
        ],
        'Value': [
            stage1['total_channels'],
            stage1['selected_channels'],
            f"{stage1['best_accuracy']:.4f}",
            f"{stage1['mean_selected_accuracy']:.4f}",
            f"{final['mean_accuracy']:.4f}",
            f"{final['std_accuracy']:.4f}",
            f"{final['best_fold_accuracy']:.4f}",
            f"{final['execution_time']:.2f}"
        ]
    }
    
    return pd.DataFrame(summary_data)
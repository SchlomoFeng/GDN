import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import time
from util.time import *
from util.env import *
from sklearn.metrics import mean_squared_error
from test import *
import torch.nn.functional as F
import numpy as np
from evaluate import get_best_performance_data, get_val_performance_data, get_full_err_scores
from sklearn.metrics import precision_score, recall_score, roc_auc_score, f1_score
from torch.utils.data import DataLoader, random_split, Subset
from scipy.stats import iqr
from util.logger import GDNLogger
from util.visualizer import GDNVisualizer




def loss_func(y_pred, y_true):
    loss = F.mse_loss(y_pred, y_true, reduction='mean')

    return loss



def train(model = None, save_path = '', config={},  train_dataloader=None, val_dataloader=None, feature_map={}, test_dataloader=None, test_dataset=None, dataset_name='swat', train_dataset=None):

    seed = config['seed']
    
    # Initialize enhanced logging and visualization
    experiment_name = f"{dataset_name}_{config.get('comment', '')}_seed{seed}"
    logger = GDNLogger(experiment_name=experiment_name, config=config)
    visualizer = GDNVisualizer(experiment_name=experiment_name)
    
    # Log model and dataset information
    dataset_info = {
        'dataset_name': dataset_name,
        'train_size': len(train_dataloader.dataset) if train_dataloader else 0,
        'val_size': len(val_dataloader.dataset) if val_dataloader else 0,
        'test_size': len(test_dataloader.dataset) if test_dataloader else 0,
        'num_features': len(feature_map),
        'feature_names': list(feature_map.keys()) if feature_map else []
    }
    logger.log_model_info(model, dataset_info)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=config['decay'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

    now = time.time()
    
    train_loss_list = []
    cmp_loss_list = []

    device = get_device()

    acu_loss = 0
    min_loss = 1e+8
    min_f1 = 0
    min_pre = 0
    best_prec = 0

    i = 0
    epoch = config['epoch']
    early_stop_win = 15

    model.train()

    log_interval = 1000
    stop_improve_count = 0

    dataloader = train_dataloader
    
    print(f"Starting training for {epoch} epochs with enhanced logging...")
    print(f"Dataset: {dataset_name}, Features: {len(feature_map)}, Device: {device}")
    print("="*60)

    for i_epoch in range(epoch):
        epoch_start_time = time.time()
        current_lr = optimizer.param_groups[0]['lr']
        
        # Log epoch start
        logger.log_epoch_start(i_epoch + 1, epoch, current_lr)

        acu_loss = 0
        batch_count = 0
        model.train()

        for batch_idx, (x, labels, attack_labels, edge_index) in enumerate(dataloader):
            _start = time.time()

            x, labels, edge_index = [item.float().to(device) for item in [x, labels, edge_index]]

            optimizer.zero_grad()
            out = model(x, edge_index).float().to(device)
            loss = loss_func(out, labels)
            
            loss.backward()
            optimizer.step()

            train_loss_list.append(loss.item())
            acu_loss += loss.item()
            batch_count += 1
                
            i += 1
            
            # Log batch progress periodically
            if batch_idx % 50 == 0 or batch_idx == len(dataloader) - 1:
                logger.log_batch_progress(batch_idx, len(dataloader), loss.item(), 'train')

        # Calculate epoch metrics
        epoch_time = time.time() - epoch_start_time
        avg_train_loss = acu_loss / len(dataloader)
        val_loss = None
        val_metrics = {}

        # Validation phase
        if val_dataloader is not None:
            print(f"Running validation for epoch {i_epoch + 1}...")
            val_loss, val_result = test(model, val_dataloader)
            
            # Calculate validation metrics
            try:
                val_scores, normal_scores = get_full_err_scores([val_result], [])
                if len(val_result) > 0 and len(val_result[2]) > 0:
                    val_labels = np.array(val_result[2])[:, 0].tolist()
                    val_metrics = get_val_performance_data(val_scores, normal_scores, val_labels, topk=1)
                    
                    logger.log_performance_metrics({
                        'val_f1': val_metrics[0] if val_metrics else 0,
                        'val_precision': val_metrics[1] if val_metrics else 0,
                        'val_recall': val_metrics[2] if val_metrics else 0
                    }, 'validation')
            except Exception as e:
                print(f"Warning: Could not calculate validation metrics: {e}")

            # Update learning rate scheduler
            scheduler.step(val_loss)

            # Early stopping and model saving logic
            if val_loss < min_loss:
                torch.save(model.state_dict(), save_path)
                min_loss = val_loss
                stop_improve_count = 0
                print(f"New best model saved with val_loss: {val_loss:.6f}")
            else:
                stop_improve_count += 1

            if stop_improve_count >= early_stop_win:
                print(f"Early stopping triggered after {early_stop_win} epochs without improvement")
                break

        else:
            # No validation data, use training loss
            if acu_loss < min_loss:
                torch.save(model.state_dict(), save_path)
                min_loss = acu_loss

        # Log epoch completion
        logger.log_epoch_end(i_epoch + 1, avg_train_loss, val_loss, epoch_time, current_lr)
        
        # Print detailed epoch summary
        print(f'Epoch {i_epoch + 1}/{epoch} Summary:')
        print(f'  Train Loss: {avg_train_loss:.6f}')
        if val_loss is not None:
            print(f'  Val Loss: {val_loss:.6f}')
        print(f'  Time: {epoch_time:.2f}s, LR: {current_lr:.6f}')
        if val_metrics:
            print(f'  Val F1: {val_metrics[0]:.4f}, Precision: {val_metrics[1]:.4f}, Recall: {val_metrics[2]:.4f}')
        print(f'  Best Val Loss: {min_loss:.6f}, Stop Count: {stop_improve_count}/{early_stop_win}')
        print("-" * 60)
        
        # Create intermediate visualizations every 10 epochs
        if (i_epoch + 1) % 10 == 0:
            try:
                training_history = logger.training_history
                visualizer.plot_training_history(training_history, save=True)
            except Exception as e:
                print(f"Warning: Could not create training visualization: {e}")

    # Training completed - save final results and create visualizations
    print("\nTraining completed!")
    training_summary = logger.get_training_summary()
    print(f"Training Summary: {training_summary}")
    
    # Save final training history and create comprehensive visualizations
    try:
        logger.save_final_results()
        print("Training history and logs saved successfully")
        
        # Create final training visualizations
        training_history = logger.training_history
        visualizer.plot_training_history(training_history, save=True)
        print("Training visualizations created successfully")
        
    except Exception as e:
        print(f"Warning: Could not save final results or create visualizations: {e}")

    return train_loss_list

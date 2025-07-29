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




def loss_func(y_pred, y_true):
    loss = F.mse_loss(y_pred, y_true, reduction='mean')

    return loss



def train(model = None, save_path = '', config={},  train_dataloader=None, val_dataloader=None, feature_map={}, test_dataloader=None, test_dataset=None, dataset_name='swat', train_dataset=None):

    seed = config['seed']
    verbose = config.get('verbose', False)
    enable_visualization = config.get('enable_visualization', False)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=config['decay'])

    now = time.time()
    
    train_loss_list = []
    val_loss_list = []
    cmp_loss_list = []
    
    # Enhanced metrics tracking
    epoch_metrics = {
        'train_loss': [],
        'val_loss': [],
        'learning_rate': [],
        'gradient_norm': []
    }

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
    
    if verbose:
        print(f"Training GDN model for {epoch} epochs...")
        print(f"  Dataset: {dataset_name}")
        print(f"  Training samples: {len(train_dataset) if train_dataset else 'Unknown'}")
        print(f"  Validation samples: {len(val_dataloader.dataset) if val_dataloader else 'None'}")
        print(f"  Batch size: {config.get('batch', 'Unknown')}")
        print(f"  Learning rate: {optimizer.param_groups[0]['lr']}")
        print(f"  Early stopping window: {early_stop_win}")
        print("-" * 60)

    for i_epoch in range(epoch):
        epoch_start_time = time.time()
        acu_loss = 0
        model.train()
        
        batch_losses = []
        gradient_norms = []

        for batch_idx, (x, labels, attack_labels, edge_index) in enumerate(dataloader):
            _start = time.time()

            x, labels, edge_index = [item.float().to(device) for item in [x, labels, edge_index]]

            optimizer.zero_grad()
            out = model(x, edge_index).float().to(device)
            loss = loss_func(out, labels)
            
            loss.backward()
            
            # Calculate gradient norm for monitoring
            total_norm = 0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** (1. / 2)
            gradient_norms.append(total_norm)
            
            optimizer.step()

            batch_losses.append(loss.item())
            train_loss_list.append(loss.item())
            acu_loss += loss.item()
                
            i += 1
            
            # Progress reporting for long epochs
            if verbose and batch_idx > 0 and batch_idx % 100 == 0:
                elapsed = time.time() - epoch_start_time
                print(f"    Epoch {i_epoch}/{epoch}, Batch {batch_idx}/{len(dataloader)}, "
                      f"Loss: {loss.item():.6f}, Time: {elapsed:.1f}s")

        # Each epoch summary
        epoch_time = time.time() - epoch_start_time
        avg_train_loss = acu_loss / len(dataloader)
        
        # Store metrics
        epoch_metrics['train_loss'].append(avg_train_loss)
        epoch_metrics['learning_rate'].append(optimizer.param_groups[0]['lr'])
        epoch_metrics['gradient_norm'].append(np.mean(gradient_norms))
        
        if verbose:
            print(f'Epoch {i_epoch:3d}/{epoch} - Train Loss: {avg_train_loss:.6f}, '
                  f'Time: {epoch_time:.1f}s, Grad Norm: {np.mean(gradient_norms):.4f}')
        else:
            print('epoch ({} / {}) (Loss:{:.8f}, ACU_loss:{:.8f})'.format(
                            i_epoch, epoch, avg_train_loss, acu_loss), flush=True)

        # Validation phase
        val_loss = None
        if val_dataloader is not None:
            val_loss, val_result = test(model, val_dataloader)
            val_loss_list.append(val_loss)
            epoch_metrics['val_loss'].append(val_loss)
            
            if verbose:
                print(f'         Validation Loss: {val_loss:.6f}')

            if val_loss < min_loss:
                torch.save(model.state_dict(), save_path)
                min_loss = val_loss
                stop_improve_count = 0
                if verbose:
                    print(f'         New best model saved (val_loss: {min_loss:.6f})')
            else:
                stop_improve_count += 1

            if stop_improve_count >= early_stop_win:
                if verbose:
                    print(f'Early stopping triggered after {i_epoch+1} epochs')
                break

        else:
            if acu_loss < min_loss :
                torch.save(model.state_dict(), save_path)
                min_loss = acu_loss
                if verbose:
                    print(f'         New best model saved (train_loss: {min_loss:.6f})')
        
        if verbose and i_epoch % 10 == 0:
            print("-" * 60)

    if verbose:
        total_time = time.time() - now
        print(f"\nTraining completed in {total_time:.1f}s ({total_time/60:.1f}m)")
        print(f"Best loss: {min_loss:.6f}")
        print(f"Final learning rate: {optimizer.param_groups[0]['lr']}")
        
        # Generate training visualization if enabled
        if enable_visualization:
            try:
                from gdn_visualizer import GDNVisualizer
                viz_dir = f'./visualizations/{dataset_name}_training'
                visualizer = GDNVisualizer(output_dir=viz_dir, verbose=False)
                
                train_data = {
                    'train_losses': train_loss_list,
                    'val_losses': val_loss_list if val_loss_list else None,
                    'metrics': epoch_metrics
                }
                
                visualizer.plot_training_curves(
                    train_data['train_losses'],
                    train_data['val_losses'],
                    train_data['metrics']
                )
                
                print(f"Training curves saved to: {viz_dir}")
                
            except Exception as e:
                print(f"Warning: Could not generate training visualizations: {e}")

    return train_loss_list

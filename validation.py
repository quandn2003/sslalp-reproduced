"""
Validation script
"""
import os
import shutil
import torch
import torch.nn as nn
import torch.optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
import torch.backends.cudnn as cudnn
import numpy as np
import matplotlib.pyplot as plt

from models.grid_proto_fewshot import FewShotSeg

from dataloaders.dev_customized_med import med_fewshot_val
from dataloaders.ManualAnnoDatasetv2 import ManualAnnoDataset
from dataloaders.GenericSuperDatasetv2 import SuperpixelDataset
from dataloaders.dataset_utils import DATASET_INFO, get_normalize_op
from dataloaders.niftiio import convert_to_sitk

from util.metric import Metric

from config_ssl_upload import ex

import tqdm
import SimpleITK as sitk
from torchvision.utils import make_grid

# config pre-trained model caching path
os.environ['TORCH_HOME'] = "./pretrained_model"


def plot_pred_gt_support(query_image, pred, gt, support_images, support_masks, score=None, save_path="debug/pred_vs_gt"):
    """
    Save 5 key images: support images, support mask, query, ground truth and prediction.
    Handles both grayscale and RGB images consistently with the same mask color.
    
    Args:
        query_image: Query image tensor (grayscale or RGB)
        pred: 2d tensor where 1 represents foreground and 0 represents background
        gt: 2d tensor where 1 represents foreground and 0 represents background
        support_images: Support image tensors (grayscale or RGB)
        support_masks: Support mask tensors
        score: Optional score to add to filename
        save_path: Base path without extension for saving images
    """
    # Create directory for this case
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Process query image - ensure HxWxC format for visualization
    query_image = query_image.clone().detach().cpu()
    if len(query_image.shape) == 3 and query_image.shape[0] <= 3:  # CHW format
        query_image = query_image.permute(1, 2, 0)
    
    # Handle grayscale vs RGB consistently
    if len(query_image.shape) == 2 or (len(query_image.shape) == 3 and query_image.shape[2] == 1):
        # For grayscale, use cmap='gray' for visualization
        is_grayscale = True
        if len(query_image.shape) == 3:
            query_image = query_image.squeeze(2)  # Remove channel dimension for grayscale
    else:
        is_grayscale = False
    
    # Normalize image for visualization
    query_image = (query_image - query_image.min()) / (query_image.max() - query_image.min() + 1e-8)
    
    # Convert pred and gt to numpy for visualization
    pred_np = pred.astype(np.float32) if isinstance(pred, np.ndarray) else pred.cpu().float().numpy()
    gt_np = gt.astype(np.float32) if isinstance(gt, np.ndarray) else gt.cpu().float().numpy()
    
    # Ensure binary masks
    pred_np = (pred_np > 0).astype(np.float32)
    gt_np = (gt_np > 0).astype(np.float32)
    
    # Set all positive values to 1.0 to ensure consistent red coloring in YlOrRd colormap
    pred_np[pred_np > 0] = 1.0
    gt_np[gt_np > 0] = 1.0
    
    # Create colormap for mask overlays - using the YlOrRd colormap as requested
    mask_cmap = plt.cm.get_cmap('YlOrRd')
    
    # Generate color masks with alpha values
    pred_rgba = mask_cmap(pred_np)
    pred_rgba[..., 3] = pred_np * 0.7  # Last channel is alpha - semitransparent where mask=1
    
    gt_rgba = mask_cmap(gt_np)
    gt_rgba[..., 3] = gt_np * 0.7  # Last channel is alpha - semitransparent where mask=1
    
    # 1. Save query image (original)
    plt.figure(figsize=(10, 10))
    if is_grayscale:
        plt.imshow(query_image, cmap='gray')
    else:
        plt.imshow(query_image)
    plt.axis('off')
    # Remove padding/whitespace
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)
    plt.savefig(f"{save_path}/query.png", bbox_inches='tight', pad_inches=0)
    plt.close()
    
    # 2. Save query image with prediction overlay
    plt.figure(figsize=(10, 10))
    if is_grayscale:
        plt.imshow(query_image, cmap='gray')
    else:
        plt.imshow(query_image)
    plt.imshow(pred_rgba)
    plt.axis('off')
    # Remove padding/whitespace
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)
    plt.savefig(f"{save_path}/pred.png", bbox_inches='tight', pad_inches=0)
    plt.close()
    
    # 3. Save query image with ground truth overlay
    plt.figure(figsize=(10, 10))
    if is_grayscale:
        plt.imshow(query_image, cmap='gray')
    else:
        plt.imshow(query_image)
    plt.imshow(gt_rgba)
    plt.axis('off')
    # Remove padding/whitespace
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)
    plt.savefig(f"{save_path}/gt.png", bbox_inches='tight', pad_inches=0)
    plt.close()
    
    # Process and save support images and masks
    if support_images is not None:
        if isinstance(support_images, list) and isinstance(support_images[0], list) and isinstance(support_images[0][0], torch.Tensor):
            support_img = support_images[0][0].clone().detach().cpu()  # Get first support image
            support_mask = support_masks[0][0].clone().detach().cpu()  # Get first support mask
            
            # Handle CHW format
            if len(support_img.shape) == 3 and support_img.shape[0] <= 3:
                support_img = support_img.permute(1, 2, 0)
                
            # Check if grayscale or RGB
            if len(support_img.shape) == 2 or (len(support_img.shape) == 3 and support_img.shape[2] == 1):
                support_is_gray = True
                if len(support_img.shape) == 3:
                    support_img = support_img.squeeze(2)
            else:
                support_is_gray = False
                
            # Normalize support image
            support_img = (support_img - support_img.min()) / (support_img.max() - support_img.min() + 1e-8)
            
            # 4. Save support image
            plt.figure(figsize=(10, 10))
            if support_is_gray:
                plt.imshow(support_img, cmap='gray')
            else:
                plt.imshow(support_img)
            plt.axis('off')
            # Remove padding/whitespace
            plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)
            plt.savefig(f"{save_path}/support_image.png", bbox_inches='tight', pad_inches=0)
            plt.close()
            
            # 5. Save support mask
            support_mask_np = support_mask.cpu().float().numpy()
            support_mask_np = (support_mask_np > 0).astype(np.float32)
            support_mask_np[support_mask_np > 0] = 1.0
            
            support_mask_rgba = mask_cmap(support_mask_np)
            support_mask_rgba[..., 3] = support_mask_np * 0.7
            
            plt.figure(figsize=(10, 10))
            if support_is_gray:
                plt.imshow(support_img, cmap='gray')
            else:
                plt.imshow(support_img)
            plt.imshow(support_mask_rgba)
            plt.axis('off')
            # Remove padding/whitespace
            plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)
            plt.savefig(f"{save_path}/support_mask.png", bbox_inches='tight', pad_inches=0)
            plt.close()


def get_dice_iou_precision_recall(pred, gt):
    """
    Calculate dice, IoU, precision and recall metrics.
    
    Args:
        pred: Binary prediction mask
        gt: Binary ground truth mask
    
    Returns:
        Dictionary with metrics
    """
    if gt.sum() == 0:
        print("gt is all background")
        return {"dice": 0, "precision": 0, "recall": 0, "iou": 0}

    tp = (pred * gt).sum()
    fp = (pred * (1 - gt)).sum()
    fn = ((1 - pred) * gt).sum()
    
    dice = 2 * tp / (2 * tp + fp + fn + 1e-8)
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    iou = tp / (tp + fp + fn + 1e-8)
    
    return {"dice": dice, "iou": iou, "precision": precision, "recall": recall}


@ex.automain
def main(_run, _config, _log):
    if _run.observers:
        os.makedirs(f'{_run.observers[0].dir}/interm_preds', exist_ok=True)
        os.makedirs(f'{_run.observers[0].dir}/visualizations', exist_ok=True)
        for source_file, _ in _run.experiment_info['sources']:
            os.makedirs(os.path.dirname(f'{_run.observers[0].dir}/source/{source_file}'),
                        exist_ok=True)
            _run.observers[0].save_file(source_file, f'source/{source_file}')
        shutil.rmtree(f'{_run.observers[0].basedir}/_sources')

    cudnn.enabled = True
    cudnn.benchmark = True
    torch.cuda.set_device(device=_config['gpu_id'])
    torch.set_num_threads(1)

    _log.info(f'###### Reload model {_config["reload_model_path"]} ######')
    model = FewShotSeg(pretrained_path = _config['reload_model_path'], cfg=_config['model'])
    model = model.cuda()
    model.eval()

    _log.info('###### Load data ######')
    ### Training set
    data_name = _config['dataset']
    if data_name == 'SABS_Superpix':
        baseset_name = 'SABS'
        max_label = 13
    elif data_name == 'C0_Superpix':
        raise NotImplementedError
        baseset_name = 'C0'
        max_label = 3
    elif data_name == 'CHAOST2_Superpix':
        baseset_name = 'CHAOST2'
        max_label = 4
    else:
        raise ValueError(f'Dataset: {data_name} not found')

    test_labels = DATASET_INFO[baseset_name]['LABEL_GROUP']['pa_all'] - DATASET_INFO[baseset_name]['LABEL_GROUP'][_config["label_sets"]]

    ### Transforms for data augmentation
    te_transforms = None

    assert _config['scan_per_load'] < 0 # by default we load the entire dataset directly

    _log.info(f'###### Labels excluded in training : {[lb for lb in _config["exclude_cls_list"]]} ######')
    _log.info(f'###### Unseen labels evaluated in testing: {[lb for lb in test_labels]} ######')

    if baseset_name == 'SABS': # for CT we need to know statistics of 
        tr_parent = SuperpixelDataset( # base dataset
            which_dataset = baseset_name,
            base_dir=_config['path'][data_name]['data_dir'],
            idx_split = _config['eval_fold'],
            mode='train',
            min_fg=str(_config["min_fg_data"]), # dummy entry for superpixel dataset
            transforms=None,
            nsup = _config['task']['n_shots'],
            scan_per_load = _config['scan_per_load'],
            exclude_list = _config["exclude_cls_list"],
            superpix_scale = _config["superpix_scale"],
            fix_length = _config["max_iters_per_load"] if (data_name == 'C0_Superpix') or (data_name == 'CHAOST2_Superpix') else None
        )
        norm_func = tr_parent.norm_func
    else:
        norm_func = get_normalize_op(modality = 'MR', fids = None)


    te_dataset, te_parent = med_fewshot_val(
        dataset_name = baseset_name,
        base_dir=_config['path'][baseset_name]['data_dir'],
        idx_split = _config['eval_fold'],
        scan_per_load = _config['scan_per_load'],
        act_labels=test_labels,
        npart = _config['task']['npart'],
        nsup = _config['task']['n_shots'],
        extern_normalize_func = norm_func
    )

    ### dataloaders
    testloader = DataLoader(
        te_dataset,
        batch_size = 1,
        shuffle=False,
        num_workers=1,
        pin_memory=False,
        drop_last=False
    )

    _log.info('###### Set validation nodes ######')
    mar_val_metric_node = Metric(max_label=max_label, n_scans= len(te_dataset.dataset.pid_curr_load) - _config['task']['n_shots'])

    _log.info('###### Starting validation ######')
    model.eval()
    mar_val_metric_node.reset()

    with torch.no_grad():
        save_pred_buffer = {} # indexed by class
        
        # Metrics per class
        class_metrics = {lb: {'dice': [], 'iou': [], 'precision': [], 'recall': []} for lb in test_labels}

        for curr_lb in test_labels:
            te_dataset.set_curr_cls(curr_lb)
            support_batched = te_parent.get_support(curr_class = curr_lb, class_idx = [curr_lb], scan_idx = _config["support_idx"], npart=_config['task']['npart'])

            # way(1 for now) x part x shot x 3 x H x W] #
            support_images = [[shot.cuda() for shot in way]
                                for way in support_batched['support_images']] # way x part x [shot x C x H x W]
            suffix = 'mask'
            support_fg_mask = [[shot[f'fg_{suffix}'].float().cuda() for shot in way]
                                for way in support_batched['support_mask']]
            support_bg_mask = [[shot[f'bg_{suffix}'].float().cuda() for shot in way]
                                for way in support_batched['support_mask']]

            curr_scan_count = -1 # counting for current scan
            _lb_buffer = {} # indexed by scan

            last_qpart = 0 # used as indicator for adding result to buffer

            for sample_batched in testloader:

                _scan_id = sample_batched["scan_id"][0] # we assume batch size for query is 1
                if _scan_id in te_parent.potential_support_sid: # skip the support scan, don't include that to query
                    continue
                if sample_batched["is_start"]:
                    ii = 0
                    curr_scan_count += 1
                    _scan_id = sample_batched["scan_id"][0]
                    outsize = te_dataset.dataset.info_by_scan[_scan_id]["array_size"]
                    outsize = (256, 256, outsize[0]) # original image read by itk: Z, H, W, in prediction we use H, W, Z
                    _pred = np.zeros( outsize )
                    _pred.fill(np.nan)

                q_part = sample_batched["part_assign"] # the chunck of query, for assignment with support
                query_images = [sample_batched['image'].cuda()]
                query_labels = torch.cat([ sample_batched['label'].cuda()], dim=0)

                # [way, [part, [shot x C x H x W]]] ->
                sup_img_part = [[shot_tensor.unsqueeze(0) for shot_tensor in support_images[0][q_part]]]   # way(1) x shot x [B(1) x C x H x W]
                sup_fgm_part = [[shot_tensor.unsqueeze(0) for shot_tensor in support_fg_mask[0][q_part]]]
                sup_bgm_part = [[shot_tensor.unsqueeze(0) for shot_tensor in support_bg_mask[0][q_part]]]

                query_pred, _, _, assign_mats = model( sup_img_part , sup_fgm_part, sup_bgm_part, query_images, isval = True, val_wsize = _config["val_wsize"] )

                query_pred_np = np.array(query_pred.argmax(dim=1)[0].cpu())
                _pred[..., ii] = query_pred_np.copy()
                
                # Calculate additional metrics
                if (sample_batched["z_id"] - sample_batched["z_max"] <= _config['z_margin']) and (sample_batched["z_id"] - sample_batched["z_min"] >= -1 * _config['z_margin']):
                    query_label_np = np.array(query_labels[0].cpu())
                    metrics = get_dice_iou_precision_recall(query_pred_np, query_label_np)
                    
                    # Record metrics
                    class_metrics[curr_lb]['dice'].append(metrics['dice'])
                    class_metrics[curr_lb]['iou'].append(metrics['iou'])
                    class_metrics[curr_lb]['precision'].append(metrics['precision'])
                    class_metrics[curr_lb]['recall'].append(metrics['recall'])
                    
                    # Log in sacred
                    _run.log_scalar(f'slice_dice_class_{curr_lb}', metrics['dice'])
                    _run.log_scalar(f'slice_iou_class_{curr_lb}', metrics['iou'])
                    _run.log_scalar(f'slice_precision_class_{curr_lb}', metrics['precision'])
                    _run.log_scalar(f'slice_recall_class_{curr_lb}', metrics['recall'])
                    
                    # Save visualization for every 5th image or for poor results
                    if ii % 5 == 0 or metrics['dice'] < 0.6:
                        viz_path = f'{_run.observers[0].dir}/visualizations/class_{curr_lb}_scan_{_scan_id}_slice_{ii}'
                        os.makedirs(viz_path, exist_ok=True)
                        plot_pred_gt_support(
                            query_images[0][0],
                            query_pred_np,
                            query_label_np, 
                            sup_img_part,
                            sup_fgm_part,
                            save_path=viz_path
                        )
                    
                    mar_val_metric_node.record(query_pred_np, query_label_np, labels=[curr_lb], n_scan=curr_scan_count) 
                else:
                    pass

                ii += 1
                # now check data format
                if sample_batched["is_end"]:
                    if _config['dataset'] != 'C0':
                        _lb_buffer[_scan_id] = _pred.transpose(2,0,1) # H, W, Z -> to Z H W
                    else:
                        lb_buffer[_scan_id] = _pred

            save_pred_buffer[str(curr_lb)] = _lb_buffer
            
            # Log average metrics for this class
            mean_dice = np.mean(class_metrics[curr_lb]['dice'])
            mean_iou = np.mean(class_metrics[curr_lb]['iou'])
            mean_precision = np.mean(class_metrics[curr_lb]['precision'])
            mean_recall = np.mean(class_metrics[curr_lb]['recall'])
            
            _run.log_scalar(f'class_{curr_lb}_mean_dice', mean_dice)
            _run.log_scalar(f'class_{curr_lb}_mean_iou', mean_iou)
            _run.log_scalar(f'class_{curr_lb}_mean_precision', mean_precision)
            _run.log_scalar(f'class_{curr_lb}_mean_recall', mean_recall)
            
            _log.info(f'Class {curr_lb} - Mean Dice: {mean_dice:.4f}, Mean IoU: {mean_iou:.4f}')
            _log.info(f'Class {curr_lb} - Mean Precision: {mean_precision:.4f}, Mean Recall: {mean_recall:.4f}')

        ### save results
        for curr_lb, _preds in save_pred_buffer.items():
            for _scan_id, _pred in _preds.items():
                _pred *= float(curr_lb)
                itk_pred = convert_to_sitk(_pred, te_dataset.dataset.info_by_scan[_scan_id])
                fid = os.path.join(f'{_run.observers[0].dir}/interm_preds', f'scan_{_scan_id}_label_{curr_lb}.nii.gz')
                sitk.WriteImage(itk_pred, fid, True)
                _log.info(f'###### {fid} has been saved ######')

        del save_pred_buffer

    del sample_batched, support_images, support_bg_mask, query_images, query_labels, query_pred

    # compute dice scores by scan
    m_classDice,_, m_meanDice,_, m_rawDice = mar_val_metric_node.get_mDice(labels=sorted(test_labels), n_scan=None, give_raw = True)

    m_classPrec,_, m_meanPrec,_,  m_classRec,_, m_meanRec,_, m_rawPrec, m_rawRec = mar_val_metric_node.get_mPrecRecall(labels=sorted(test_labels), n_scan=None, give_raw = True)

    mar_val_metric_node.reset() # reset this calculation node

    # write validation result to log file
    _run.log_scalar('mar_val_batches_classDice', m_classDice.tolist())
    _run.log_scalar('mar_val_batches_meanDice', m_meanDice.tolist())
    _run.log_scalar('mar_val_batches_rawDice', m_rawDice.tolist())

    _run.log_scalar('mar_val_batches_classPrec', m_classPrec.tolist())
    _run.log_scalar('mar_val_batches_meanPrec', m_meanPrec.tolist())
    _run.log_scalar('mar_val_batches_rawPrec', m_rawPrec.tolist())

    _run.log_scalar('mar_val_batches_classRec', m_classRec.tolist())
    _run.log_scalar('mar_val_al_batches_meanRec', m_meanRec.tolist())
    _run.log_scalar('mar_val_al_batches_rawRec', m_rawRec.tolist())

    _log.info(f'mar_val batches classDice: {m_classDice}')
    _log.info(f'mar_val batches meanDice: {m_meanDice}')

    _log.info(f'mar_val batches classPrec: {m_classPrec}')
    _log.info(f'mar_val batches meanPrec: {m_meanPrec}')

    _log.info(f'mar_val batches classRec: {m_classRec}')
    _log.info(f'mar_val batches meanRec: {m_meanRec}')

    print("============ ============")

    # Calculate overall metrics across all classes
    overall_metrics = {
        'dice': np.mean([np.mean(class_metrics[lb]['dice']) for lb in test_labels]),
        'iou': np.mean([np.mean(class_metrics[lb]['iou']) for lb in test_labels]),
        'precision': np.mean([np.mean(class_metrics[lb]['precision']) for lb in test_labels]),
        'recall': np.mean([np.mean(class_metrics[lb]['recall']) for lb in test_labels])
    }
    
    _run.log_scalar('overall_mean_dice', overall_metrics['dice'])
    _run.log_scalar('overall_mean_iou', overall_metrics['iou'])
    _run.log_scalar('overall_mean_precision', overall_metrics['precision'])
    _run.log_scalar('overall_mean_recall', overall_metrics['recall'])
    
    _log.info(f"Overall Mean Dice: {overall_metrics['dice']:.4f}")
    _log.info(f"Overall Mean IoU: {overall_metrics['iou']:.4f}")
    _log.info(f"Overall Mean Precision: {overall_metrics['precision']:.4f}")
    _log.info(f"Overall Mean Recall: {overall_metrics['recall']:.4f}")

    _log.info(f'End of validation')
    return 1



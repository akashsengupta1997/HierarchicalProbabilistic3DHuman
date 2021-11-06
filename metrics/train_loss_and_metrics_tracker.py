import numpy as np
import pickle

from utils.eval_utils import procrustes_analysis_batch, scale_and_translation_transform_batch
from utils.joints2d_utils import undo_keypoint_normalisation


class TrainingLossesAndMetricsTracker:
    """
    Tracks training and validation losses (both total and per-task) and metrics during
    training. Updates loss and metrics history at end of each epoch.
    """
    def __init__(self,
                 metrics_to_track,
                 img_wh,
                 log_save_path,
                 load_logs=False,
                 current_epoch=None):

        self.all_metrics_types = ['train_PVE', 'val_PVE',
                                  'train_PVE-SC', 'val_PVE-SC',
                                  'train_PVE-PA', 'val_PVE-PA',
                                  'train_PVE-T', 'val_PVE-T',
                                  'train_PVE-T-SC', 'val_PVE-T-SC',
                                  'train_MPJPE', 'val_MPJPE',
                                  'train_MPJPE-SC', 'val_MPJPE-SC',
                                  'train_MPJPE-PA', 'val_MPJPE-PA',
                                  'train_joints2D-L2E', 'val_joints2D-L2E',
                                  'train_joints2Dsamples-L2E', 'val_joints2Dsamples-L2E']

        self.metrics_to_track = metrics_to_track
        self.img_wh = img_wh
        self.log_save_path = log_save_path

        if load_logs:
            self.epochs_history = self.load_history(log_save_path, current_epoch)
        else:
            self.epochs_history = {'train_losses': [], 'val_losses': []}
            for metric_type in self.all_metrics_types:
                self.epochs_history[metric_type] = []

        self.loss_metric_sums = None

    def load_history(self, load_log_path, current_epoch):
        """
        Loads loss (total and per-task) and metric history up to current epoch given the path
        to a log file. If per-task losses or metrics are missing from log file, fill with 0s.
        """
        with open(load_log_path, 'rb') as f:
            history = pickle.load(f)

        history['train_losses'] = history['train_losses'][:current_epoch]
        history['val_losses'] = history['val_losses'][:current_epoch]

        # For each metric, if metric in history, load up to current epoch.
        # Else, fill with 0s up to current epoch.
        for metric_type in self.all_metrics_types:
            if metric_type in history.keys():
                history[metric_type] = history[metric_type][:current_epoch]
            else:
                history[metric_type] = [0.0] * current_epoch
                print(metric_type, 'filled with zeros up to epoch', current_epoch)

        for key in history.keys():
            assert len(history[key]) == current_epoch, \
                "{} elements in {} list when current epoch is {}".format(
                    str(len(history[key])),
                    key,
                    str(current_epoch))
        print('Logs loaded from', load_log_path)

        return history

    def initialise_loss_metric_sums(self):
        self.loss_metric_sums = {'train_losses': 0., 'val_losses': 0.,
                                 'train_num_samples': 0, 'val_num_samples': 0}

        for metric_type in self.all_metrics_types:
            if metric_type == 'train_joints2Dsamples-L2E':
                self.loss_metric_sums['train_num_visib_joints2Dsamples'] = 0.
                self.loss_metric_sums[metric_type] = 0.
            elif metric_type == 'val_joints2Dsamples-L2E':
                self.loss_metric_sums['val_num_visib_joints2Dsamples'] = 0.
                self.loss_metric_sums[metric_type] = 0.
            else:
                self.loss_metric_sums[metric_type] = 0.

    def update_per_batch(self,
                         split,
                         loss,
                         pred_dict,
                         target_dict,
                         batch_size,
                         pred_reposed_vertices=None,
                         target_reposed_vertices=None):
        assert split in ['train', 'val'], "Invalid split in metric tracker batch update."

        if any(['PVE-T' in metric_type for metric_type in self.metrics_to_track]):
            assert (pred_reposed_vertices is not None) and \
                   (target_reposed_vertices is not None), \
                "Need to pass reposed vertices to metric tracker batch update."
            pred_reposed_vertices = pred_reposed_vertices.cpu().detach().numpy()
            target_reposed_vertices = target_reposed_vertices.cpu().detach().numpy()

        for key in pred_dict.keys():
            pred_dict[key] = pred_dict[key].cpu().detach().numpy()
            if key in target_dict.keys():
                target_dict[key] = target_dict[key].cpu().detach().numpy()

        if 'joints2Dsamples-L2E' in self.metrics_to_track:
            target_dict['joints2D_vis'] = target_dict['joints2D_vis'].cpu().detach().numpy()

        # -------- Update loss sums --------
        self.loss_metric_sums[split + '_losses'] += loss.item() * batch_size
        self.loss_metric_sums[split + '_num_samples'] += batch_size

        # -------- Update metrics sums --------
        if 'PVE' in self.metrics_to_track:
            pve_batch = np.linalg.norm(pred_dict['verts'] - target_dict['verts'], axis=-1)  # (bsize, 6890) or (bsize, num views, 6890)
            self.loss_metric_sums[split + '_PVE'] += np.sum(pve_batch)  # scalar

        # Scale and translation correction
        if 'PVE-SC' in self.metrics_to_track:
            pred_vertices = pred_dict['verts']  # (bsize, 6890, 3) or (bsize, num views, 6890, 3)
            target_vertices = target_dict['verts']  # (bsize, 6890, 3) or (bsize, num views, 6890, 3)
            pred_vertices_sc = scale_and_translation_transform_batch(pred_vertices.reshape(-1, 6890, 3),
                                                                     target_vertices.reshape(-1, 6890, 3))
            pve_sc_batch = np.linalg.norm(pred_vertices_sc - target_vertices.reshape(-1, 6890, 3), axis=-1)  # (bsize, 6890) or (bsize*num views, 6890)
            self.loss_metric_sums[split + '_PVE-SC'] += np.sum(pve_sc_batch)  # scalar

        # Procrustes analysis
        if 'PVE-PA' in self.metrics_to_track:
            pred_vertices = pred_dict['verts']  # (bsize, 6890, 3)  or (bsize, num views, 6890, 3)
            target_vertices = target_dict['verts']  # (bsize, 6890, 3) or (bsize, num views, 6890, 3)
            pred_vertices_pa = procrustes_analysis_batch(pred_vertices.reshape(-1, 6890, 3),
                                                         target_vertices.reshape(-1, 6890, 3))
            pve_pa_batch = np.linalg.norm(pred_vertices_pa - target_vertices.reshape(-1, 6890, 3), axis=-1)  # (bsize, 6890) or (bsize*num views, 6890)
            self.loss_metric_sums[split + '_PVE-PA'] += np.sum(pve_pa_batch)  # scalar

        # Reposed
        if 'PVE-T' in self.metrics_to_track:
            pvet_batch = np.linalg.norm(pred_reposed_vertices - target_reposed_vertices, axis=-1)
            self.loss_metric_sums[split + '_PVE-T'] += np.sum(pvet_batch)

        # Reposed + Scale and translation correction
        if 'PVE-T-SC' in self.metrics_to_track:
            pred_reposed_vertices_sc = scale_and_translation_transform_batch(pred_reposed_vertices,
                                                                             target_reposed_vertices)
            pvet_sc_batch = np.linalg.norm(pred_reposed_vertices_sc - target_reposed_vertices, axis=-1)  # (bs, 6890)
            self.loss_metric_sums[split + '_PVE-T-SC'] += np.sum(pvet_sc_batch)  # scalar

        if 'MPJPE' in self.metrics_to_track:
            mpjpe_batch = np.linalg.norm(pred_dict['joints3D'] - target_dict['joints3D'], axis=-1)  # (bsize, 14) or (bsize, num views, 14)
            self.loss_metric_sums[split + '_MPJPE'] += np.sum(mpjpe_batch)  # scalar

        # Scale and translation correction
        if 'MPJPE-SC' in self.metrics_to_track:
            pred_joints3D_h36mlsp = pred_dict['joints3D']  # (bsize, 14, 3) or (bsize, num views, 14, 3)
            target_joints3D_h36mlsp = target_dict['joints3D']  # (bsize, 14, 3) or (bsize, num views, 14, 3)
            pred_joints3D_h36mlsp_sc = scale_and_translation_transform_batch(pred_joints3D_h36mlsp.reshape(-1, 14, 3),
                                                                             target_joints3D_h36mlsp.reshape(-1, 14, 3))
            mpjpe_sc_batch = np.linalg.norm(pred_joints3D_h36mlsp_sc - target_joints3D_h36mlsp.reshape(-1, 14, 3),
                                            axis=-1)  # (bsize, 14) or (bsize*num views, 14)
            self.loss_metric_sums[split + '_MPJPE-SC'] += np.sum(mpjpe_sc_batch)  # scalar

        # Procrustes analysis
        if 'MPJPE-PA' in self.metrics_to_track:
            pred_joints3D_h36mlsp = pred_dict['joints3D']  # (bsize, 14, 3) or (bsize, num views, 14, 3)
            target_joints3D_h36mlsp = target_dict['joints3D']  # (bsize, 14, 3) or (bsize, num views, 14, 3)
            pred_joints3D_h36mlsp_pa = procrustes_analysis_batch(pred_joints3D_h36mlsp.reshape(-1, 14, 3),
                                                                 target_joints3D_h36mlsp.reshape(-1, 14, 3))
            mpjpe_pa_batch = np.linalg.norm(pred_joints3D_h36mlsp_pa - target_joints3D_h36mlsp.reshape(-1, 14, 3),
                                            axis=-1)  # (bsize, 14) or (bsize*num views, 14)
            self.loss_metric_sums[split + '_MPJPE-PA'] += np.sum(mpjpe_pa_batch)  # scalar

        if 'joints2D-L2E' in self.metrics_to_track:
            pred_joints2D_coco = pred_dict['joints2D'] # (bsize, 17, 2) or (bsize, num views, 17, 2)
            target_joints2D_coco = target_dict['joints2D']  # (bsize, 17, 2) or (bsize, num views, 17, 2)
            pred_joints2D_coco = undo_keypoint_normalisation(pred_joints2D_coco, self.img_wh)
            joints2D_l2e_batch = np.linalg.norm(pred_joints2D_coco - target_joints2D_coco, axis=-1)  # (bsize, 17) or (bsize, num views, 17)
            self.loss_metric_sums[split + '_joints2D-L2E'] += np.sum(joints2D_l2e_batch)  # scalar

        if 'joints2Dsamples-L2E' in self.metrics_to_track:
            pred_joints2D_coco_samples = pred_dict['joints2Dsamples']  # (bsize, num_samples, 17, 2)

            target_joints2D_coco = np.tile(target_dict['joints2D'][:, None, :, :], (1, pred_joints2D_coco_samples.shape[1], 1, 1))  # (bsize, num_samples, 17, 2)
            target_joints2d_vis_coco = np.tile(target_dict['joints2D_vis'][:, None, :], (1, pred_joints2D_coco_samples.shape[1], 1))   # (bsize, num_samples, 17)
            target_joints2D_coco = target_joints2D_coco[target_joints2d_vis_coco, :]  # (N, 2)

            pred_joints2D_coco_samples = pred_joints2D_coco_samples[target_joints2d_vis_coco, :]  # (N, 2)
            pred_joints2D_coco_samples = undo_keypoint_normalisation(pred_joints2D_coco_samples, self.img_wh)

            joints2Dsamples_l2e_batch = np.linalg.norm(pred_joints2D_coco_samples - target_joints2D_coco, axis=-1)  # (N,)
            assert joints2Dsamples_l2e_batch.shape[0] == target_joints2d_vis_coco.sum()
            self.loss_metric_sums[split + '_joints2Dsamples-L2E'] += np.sum(joints2Dsamples_l2e_batch)  # scalar
            self.loss_metric_sums[split + '_num_visib_joints2Dsamples'] += joints2Dsamples_l2e_batch.shape[0]

    def update_per_epoch(self):
        self.epochs_history['train_losses'].append(self.loss_metric_sums['train_losses'] / self.loss_metric_sums['train_num_samples'])
        self.epochs_history['val_losses'].append(self.loss_metric_sums['val_losses'] / self.loss_metric_sums['val_num_samples'])

        # For each metric, if tracking metric, append metric per sample for current epoch to
        # loss history. Else, append 0.
        for metric_type in self.all_metrics_types:
            split = metric_type.split('_')[0]

            if metric_type[metric_type.find('_')+1:] in self.metrics_to_track:
                if 'joints2Dsamples' in metric_type:
                    joints2Dsamples_l2e = self.loss_metric_sums[split + '_joints2Dsamples-L2E'] / self.loss_metric_sums[split+'_num_visib_joints2Dsamples']
                    self.epochs_history[metric_type].append(joints2Dsamples_l2e)
                else:
                    if 'PVE' in metric_type:
                        num_per_sample = 6890  # num 3D verts per sample
                    elif 'MPJPE' in metric_type:
                        num_per_sample = 14  # num 2D joints per sample
                    elif 'joints2D' in metric_type:
                        num_per_sample = 17  # num 3D joints per sample

                    self.epochs_history[metric_type].append(self.loss_metric_sums[metric_type] / (self.loss_metric_sums[split + '_num_samples'] * num_per_sample))
            else:
                self.epochs_history[metric_type].append(0.)

        # Print end of epoch losses and metrics.
        print('Finished epoch.')
        print('Train Loss: {:.5f}, Val Loss: {:.5f}'.format(self.epochs_history['train_losses'][-1],
                                                            self.epochs_history['val_losses'][-1]))
        for metric in self.metrics_to_track:
            print('Train {}: {:.5f}, Val {}: {:.5f}'.format(metric,
                                                            self.epochs_history['train_' + metric][-1],
                                                            metric,
                                                            self.epochs_history['val_' + metric][-1]))

        # Dump history to log file
        if self.log_save_path is not None:
            with open(self.log_save_path, 'wb') as f_out:
                pickle.dump(self.epochs_history, f_out)

    def determine_save_model_weights_this_epoch(self, save_val_metrics, best_epoch_val_metrics):
        save_model_weights_this_epoch = True
        for metric in save_val_metrics:
            if self.epochs_history['val_' + metric][-1] > best_epoch_val_metrics[metric]:
                save_model_weights_this_epoch = False
                break

        return save_model_weights_this_epoch


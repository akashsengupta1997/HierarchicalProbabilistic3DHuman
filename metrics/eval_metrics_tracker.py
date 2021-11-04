import numpy as np
import os

from utils.eval_utils import procrustes_analysis_batch, scale_and_translation_transform_batch


class EvalMetricsTracker:
    """
    Tracks metrics during evaluation.
    Mode metric types:
        - PVE: 
            - Per vertex 3D position error (L2 norm in millimetres) (should really rename this to MPVPE)
        - PVE-T: 
            - Per vertex 3D position error in the T-pose/neutral pose i.e. body shape error (L2 norm in millimetres)
        - MPJPE: 
            - Per joint 3D position error (L2 norm in millimetres)
        - Joints2D L2E: 
            - Per joint 2D position error after projection to image plane (L2 norm in pixels)
            - If the visibility of target 2D joint is provided, this metric will be computed using only visible targets
        - Silhouette IOU:
            - Silhouette intersection over union
    Sample metric Types:
        - PVE/PVE-T/MPJPE Samples Minimum:
            - Minimum 3D position error out of N samples obtained from shape/pose distribution.
        - Joints2D Samples L2E:
            - Mean per joint 2D position error over N samples obtained from shape/pose distribution.
            - All 2D samples from predicted 3D distribution should match 2D target joints.
        - Silhouette samples IOU
            - Mean silhouette-IOU over N samples obtained from shape/pose distribution.
            - All 2D samples from predicted 3D distribution should match 2D target silhouette.
    """
    def __init__(self,
                 metrics_to_track,
                 img_wh=None,
                 save_path=None,
                 save_per_frame_metrics=False):

        self.metrics_to_track = metrics_to_track
        self.img_wh = img_wh

        self.metric_sums = None
        self.total_samples = 0
        self.save_per_frame_metrics = save_per_frame_metrics
        self.save_path = save_path
        print('\nInitialised metrics tracker.')

    def initialise_metric_sums(self):
        self.metric_sums = {}
        for metric_type in self.metrics_to_track:
            if metric_type == 'silhouette-IOU':
                self.metric_sums['num_true_positives'] = 0.
                self.metric_sums['num_false_positives'] = 0.
                self.metric_sums['num_true_negatives'] = 0.
                self.metric_sums['num_false_negatives'] = 0.
            elif metric_type == 'silhouettesamples-IOU':
                self.metric_sums['num_samples_true_positives'] = 0.
                self.metric_sums['num_samples_false_positives'] = 0.
                self.metric_sums['num_samples_true_negatives'] = 0.
                self.metric_sums['num_samples_false_negatives'] = 0.
            elif metric_type == 'joints2Dsamples-L2E':
                self.metric_sums['num_vis_joints2Dsamples'] = 0.
                self.metric_sums[metric_type] = 0.
            else:
                self.metric_sums[metric_type] = 0.

    def initialise_per_frame_metric_lists(self):
        self.per_frame_metrics = {}
        for metric_type in self.metrics_to_track:
            self.per_frame_metrics[metric_type] = []

    def update_per_batch(self,
                         pred_dict,
                         target_dict,
                         num_input_samples,
                         return_transformed_points=False,
                         return_per_frame_metrics=False):
        self.total_samples += num_input_samples

        if return_transformed_points:
            transformed_points_return_dict = {}
        else:
            transformed_points_return_dict = None
        if return_per_frame_metrics:
            per_frame_metrics_return_dict = {}
        else:
            per_frame_metrics_return_dict = None

        # -------- Update metrics sums --------
        if 'PVE' in self.metrics_to_track:
            pve_batch = np.linalg.norm(pred_dict['verts'] - target_dict['verts'], axis=-1)  # (bsize, 6890) or (num views, 6890)
            self.metric_sums['PVE'] += np.sum(pve_batch)  # scalar
            self.per_frame_metrics['PVE'].append(np.mean(pve_batch, axis=-1))  # (bs,) or (num views,)
            if return_per_frame_metrics:
                per_frame_metrics_return_dict['PVE'] = np.mean(pve_batch, axis=-1)

        # Scale and translation correction
        if 'PVE-SC' in self.metrics_to_track:
            pred_vertices = pred_dict['verts']  # (bsize, 6890, 3) or (num views, 6890, 3)
            target_vertices = target_dict['verts']  # (bsize, 6890, 3) or (num views, 6890, 3)
            pred_vertices_sc = scale_and_translation_transform_batch(pred_vertices, target_vertices)
            pve_sc_batch = np.linalg.norm(pred_vertices_sc - target_vertices, axis=-1)  # (bs, 6890) or (num views, 6890)
            self.metric_sums['PVE-SC'] += np.sum(pve_sc_batch)  # scalar
            self.per_frame_metrics['PVE-SC'].append(np.mean(pve_sc_batch, axis=-1))  # (bs,) or (num views,)
            if return_transformed_points:
                transformed_points_return_dict['pred_vertices_sc'] = pred_vertices_sc
            if return_per_frame_metrics:
                per_frame_metrics_return_dict['PVE-SC'] = np.mean(pve_sc_batch, axis=-1)

        # Procrustes analysis
        if 'PVE-PA' in self.metrics_to_track:
            pred_vertices = pred_dict['verts']  # (bsize, 6890, 3) or (num views, 6890, 3)
            target_vertices = target_dict['verts']  # (bsize, 6890, 3) or (num views, 6890, 3)
            pred_vertices_pa = procrustes_analysis_batch(pred_vertices, target_vertices)
            pve_pa_batch = np.linalg.norm(pred_vertices_pa - target_vertices, axis=-1)  # (bsize, 6890) or (num views, 6890)
            self.metric_sums['PVE-PA'] += np.sum(pve_pa_batch)  # scalar
            self.per_frame_metrics['PVE-PA'].append(np.mean(pve_pa_batch, axis=-1))  # (bs,) or (num views,)
            if return_transformed_points:
                transformed_points_return_dict['pred_vertices_pa'] = pred_vertices_pa
            if return_per_frame_metrics:
                per_frame_metrics_return_dict['PVE-PA'] = np.mean(pve_pa_batch, axis=-1)

        # Reposed
        if 'PVE-T' in self.metrics_to_track:
            pvet_batch = np.linalg.norm(pred_dict['reposed_verts'] - target_dict['reposed_verts'], axis=-1)  # (bsize, 6890) or (num views, 6890)
            self.metric_sums['PVE-T'] += np.sum(pvet_batch)  # scalar
            self.per_frame_metrics['PVE-T'].append(np.mean(pvet_batch, axis=-1))  # (bs,)
            if return_per_frame_metrics:
                per_frame_metrics_return_dict['PVE-T'] = np.mean(pvet_batch, axis=-1)

        # Reposed + Scale and translation correction
        if 'PVE-T-SC' in self.metrics_to_track:
            pred_reposed_vertices = pred_dict['reposed_verts']  # (bsize, 6890, 3) or (num views, 6890, 3)
            target_reposed_vertices = target_dict['reposed_verts']  # (bsize, 6890, 3) or (num views, 6890, 3)
            pred_reposed_vertices_sc = scale_and_translation_transform_batch(pred_reposed_vertices,
                                                                             target_reposed_vertices)
            pvet_sc_batch = np.linalg.norm(pred_reposed_vertices_sc - target_reposed_vertices, axis=-1)  # (bs, 6890) or (num views, 6890)
            self.metric_sums['PVE-T-SC'] += np.sum(pvet_sc_batch)  # scalar
            self.per_frame_metrics['PVE-T-SC'].append(np.mean(pvet_sc_batch, axis=-1))  # (bs,) or (num views,)
            if return_transformed_points:
                transformed_points_return_dict['pred_reposed_vertices_sc'] = pred_reposed_vertices_sc
            if return_per_frame_metrics:
                per_frame_metrics_return_dict['PVE-T-SC'] = np.mean(pvet_sc_batch, axis=-1)

        if 'MPJPE' in self.metrics_to_track:
            mpjpe_batch = np.linalg.norm(pred_dict['joints3D'] - target_dict['joints3D'], axis=-1)  # (bsize, 14) or (num views, 14)
            self.metric_sums['MPJPE'] += np.sum(mpjpe_batch)  # scalar
            self.per_frame_metrics['MPJPE'].append(np.mean(mpjpe_batch, axis=-1))  # (bs,) or (num views,)
            if return_per_frame_metrics:
                per_frame_metrics_return_dict['MPJPE'] = np.mean(mpjpe_batch, axis=-1)

        # Scale and translation correction
        if 'MPJPE-SC' in self.metrics_to_track:
            pred_joints3D_h36mlsp = pred_dict['joints3D']  # (bsize, 14, 3) or (num views, 14, 3)
            target_joints3D_h36mlsp = target_dict['joints3D']  # (bsize, 14, 3) or (num views, 14, 3)
            pred_joints3D_h36mlsp_sc = scale_and_translation_transform_batch(pred_joints3D_h36mlsp,
                                                                             target_joints3D_h36mlsp)
            mpjpe_sc_batch = np.linalg.norm(pred_joints3D_h36mlsp_sc - target_joints3D_h36mlsp, axis=-1)  # (bsize, 14) or (num views, 14)
            self.metric_sums['MPJPE-SC'] += np.sum(mpjpe_sc_batch)  # scalar
            self.per_frame_metrics['MPJPE-SC'].append(np.mean(mpjpe_sc_batch, axis=-1))  # (bs,) or (num views,)
            if return_transformed_points:
                transformed_points_return_dict['pred_joints3D_h36mlsp_sc'] = pred_joints3D_h36mlsp_sc
            if return_per_frame_metrics:
                per_frame_metrics_return_dict['MPJPE-SC'] = np.mean(mpjpe_sc_batch, axis=-1)

        # Procrustes analysis
        if 'MPJPE-PA' in self.metrics_to_track:
            pred_joints3D_h36mlsp = pred_dict['joints3D']  # (bsize, 14, 3) or (num views, 14, 3)
            target_joints3D_h36mlsp = target_dict['joints3D']  # (bsize, 14, 3) or (num views, 14, 3)
            pred_joints3D_h36mlsp_pa = procrustes_analysis_batch(pred_joints3D_h36mlsp,
                                                                 target_joints3D_h36mlsp)
            mpjpe_pa_batch = np.linalg.norm(pred_joints3D_h36mlsp_pa - target_joints3D_h36mlsp, axis=-1)  # (bsize, 14) or (num views, 14)
            self.metric_sums['MPJPE-PA'] += np.sum(mpjpe_pa_batch)  # scalar
            self.per_frame_metrics['MPJPE-PA'].append(np.mean(mpjpe_pa_batch, axis=-1))  # (bs,) or (num views,)
            if return_transformed_points:
                transformed_points_return_dict['pred_joints3D_h36mlsp_pa'] = pred_joints3D_h36mlsp_pa
            if return_per_frame_metrics:
                per_frame_metrics_return_dict['MPJPE-PA'] = np.mean(mpjpe_pa_batch, axis=-1)

        if 'PVE_samples_min' in self.metrics_to_track:
            assert num_input_samples == 1, "Batch size must be 1 for min samples metrics!"
            pve_per_sample = np.linalg.norm(pred_dict['verts_samples'] - target_dict['verts'], axis=-1)  # (num samples, 6890)
            min_pve_sample = np.argmin(np.mean(pve_per_sample, axis=-1))
            pve_samples_min_batch = pve_per_sample[min_pve_sample]
            self.metric_sums['PVE_samples_min'] += np.sum(pve_samples_min_batch)  # scalar
            self.per_frame_metrics['PVE_samples_min'].append(np.mean(pve_samples_min_batch, axis=-1))  # (1,) i.e. scalar

        # Scale and translation correction
        if 'PVE-SC_samples_min' in self.metrics_to_track:
            assert num_input_samples == 1, "Batch size must be 1 for min samples metrics!"
            pred_vertices_samples = pred_dict['verts_samples']  # (num samples, 6890, 3)
            target_vertices = np.tile(target_dict['verts'], (pred_vertices_samples.shape[0], 1, 1))  # (num samples, 6890, 3)
            pred_vertices_samples_sc = scale_and_translation_transform_batch(pred_vertices_samples, target_vertices)
            pve_sc_per_sample = np.linalg.norm(pred_vertices_samples_sc - target_vertices, axis=-1)  # (num samples, 6890)
            min_pve_sc_sample = np.argmin(np.mean(pve_sc_per_sample, axis=-1))
            pve_sc_samples_min_batch = pve_sc_per_sample[min_pve_sc_sample]
            self.metric_sums['PVE-SC_samples_min'] += np.sum(pve_sc_samples_min_batch)  # scalar
            self.per_frame_metrics['PVE-SC_samples_min'].append(np.mean(pve_sc_samples_min_batch, axis=-1))  # (1,) i.e. scalar

        # Procrustes analysis
        if 'PVE-PA_samples_min' in self.metrics_to_track:
            assert num_input_samples == 1, "Batch size must be 1 for min samples metrics!"
            pred_vertices_samples = pred_dict['verts_samples']  # (num samples, 6890, 3)
            target_vertices = np.tile(target_dict['verts'], (pred_vertices_samples.shape[0], 1, 1))  # (num samples, 6890, 3)
            pred_vertices_samples_pa = procrustes_analysis_batch(pred_vertices_samples, target_vertices)
            pve_pa_per_sample = np.linalg.norm(pred_vertices_samples_pa - target_vertices, axis=-1)  # (num samples, 6890)
            min_pve_pa_sample = np.argmin(np.mean(pve_pa_per_sample, axis=-1))
            pve_pa_samples_min_batch = pve_pa_per_sample[min_pve_pa_sample]
            self.metric_sums['PVE-PA_samples_min'] += np.sum(pve_pa_samples_min_batch)  # scalar
            self.per_frame_metrics['PVE-PA_samples_min'].append(np.mean(pve_pa_samples_min_batch, axis=-1))  # (1,) i.e. scalar

        # Reposed
        if 'PVE-T_samples_min' in self.metrics_to_track:
            assert num_input_samples == 1, "Batch size must be 1 for min samples metrics!"
            pvet_per_sample = np.linalg.norm(pred_dict['reposed_verts_samples'] - target_dict['reposed_verts'],
                                             axis=-1)  # (num samples, 6890)
            min_pvet_sample = np.argmin(np.mean(pvet_per_sample, axis=-1))
            pvet_samples_min_batch = pvet_per_sample[min_pvet_sample]
            self.metric_sums['PVE-T_samples_min'] += np.sum(pvet_samples_min_batch)  # scalar
            self.per_frame_metrics['PVE-T_samples_min'].append(np.mean(pvet_samples_min_batch, axis=-1))  # (1,) i.e. scalar

        # Reposed + Scale and translation correction
        if 'PVE-T-SC_samples_min' in self.metrics_to_track:
            assert num_input_samples == 1, "Batch size must be 1 for min samples metrics!"
            pred_reposed_vertices_samples = pred_dict['reposed_verts_samples']  # (num samples, 6890, 3)
            target_reposed_vertices = np.tile(target_dict['reposed_verts'],
                                              (pred_reposed_vertices_samples.shape[0], 1, 1))  # (num samples, 6890, 3)
            pred_reposed_vertices_samples_sc = scale_and_translation_transform_batch(pred_reposed_vertices_samples,
                                                                                     target_reposed_vertices)
            pvet_sc_per_sample = np.linalg.norm(pred_reposed_vertices_samples_sc - target_reposed_vertices, axis=-1)  # (num samples, 6890)
            min_pvet_sc_sample = np.argmin(np.mean(pvet_sc_per_sample, axis=-1))
            pvet_sc_samples_min_batch = pvet_sc_per_sample[min_pvet_sc_sample]
            self.metric_sums['PVE-T-SC_samples_min'] += np.sum(pvet_sc_samples_min_batch)  # scalar
            self.per_frame_metrics['PVE-T-SC_samples_min'].append(np.mean(pvet_sc_samples_min_batch, axis=-1))  # (1,) i.e. scalar

        if 'MPJPE_samples_min' in self.metrics_to_track:
            assert num_input_samples == 1, "Batch size must be 1 for min samples metrics!"
            mpjpe_per_sample = np.linalg.norm(pred_dict['joints3D_samples'] - target_dict['joints3D'], axis=-1)  # (num samples, 14))
            min_mpjpe_sample = np.argmin(np.mean(mpjpe_per_sample, axis=-1))
            mpjpe_samples_min_batch = mpjpe_per_sample[min_mpjpe_sample]
            self.metric_sums['MPJPE_samples_min'] += np.sum(mpjpe_samples_min_batch)  # scalar
            self.per_frame_metrics['MPJPE_samples_min'].append(np.mean(mpjpe_samples_min_batch, axis=-1))  # (1,) i.e. scalar

        # Scale and translation correction
        if 'MPJPE-SC_samples_min' in self.metrics_to_track:
            assert num_input_samples == 1, "Batch size must be 1 for min samples metrics!"
            pred_joints3D_h36mlsp_samples = pred_dict['joints3D_samples']  # (num samples, 14, 3)
            target_joints3D_h36mlsp = np.tile(target_dict['joints3D'],
                                              (pred_joints3D_h36mlsp_samples.shape[0], 1, 1))  # (num samples, 14, 3)
            pred_joints3D_h36mlsp_sc = scale_and_translation_transform_batch(pred_joints3D_h36mlsp_samples,
                                                                             target_joints3D_h36mlsp)
            mpjpe_sc_per_sample = np.linalg.norm(pred_joints3D_h36mlsp_sc - target_joints3D_h36mlsp, axis=-1)  # (num samples, 14)
            min_mpjpe_sc_sample = np.argmin(np.mean(mpjpe_sc_per_sample, axis=-1))
            mpjpe_sc_samples_min_batch = mpjpe_sc_per_sample[min_mpjpe_sc_sample]
            self.metric_sums['MPJPE-SC_samples_min'] += np.sum(mpjpe_sc_samples_min_batch)  # scalar
            self.per_frame_metrics['MPJPE-SC_samples_min'].append(np.mean(mpjpe_sc_samples_min_batch, axis=-1))  # (1,) i.e. scalar

        # Procrustes analysis
        if 'MPJPE-PA_samples_min' in self.metrics_to_track:
            assert num_input_samples == 1, "Batch size must be 1 for min samples metrics!"
            pred_joints3D_h36mlsp_samples = pred_dict['joints3D_samples']  # (num samples, 14, 3)
            target_joints3D_h36mlsp = np.tile(target_dict['joints3D'],
                                              (pred_joints3D_h36mlsp_samples.shape[0], 1, 1))  # (num samples, 14, 3)
            pred_joints3D_h36mlsp_pa = procrustes_analysis_batch(pred_joints3D_h36mlsp_samples,
                                                                 target_joints3D_h36mlsp)
            mpjpe_pa_per_sample = np.linalg.norm(pred_joints3D_h36mlsp_pa - target_joints3D_h36mlsp, axis=-1)  # (num samples, 14)
            min_mpjpe_pa_sample = np.argmin(np.mean(mpjpe_pa_per_sample, axis=-1))
            mpjpe_pa_samples_min_batch = mpjpe_pa_per_sample[min_mpjpe_pa_sample]
            self.metric_sums['MPJPE-PA_samples_min'] += np.sum(mpjpe_pa_samples_min_batch)  # scalar
            self.per_frame_metrics['MPJPE-PA_samples_min'].append(np.mean(mpjpe_pa_samples_min_batch, axis=-1))  # (1,) i.e. scalar

        if 'joints2D-L2E' in self.metrics_to_track:
            pred_joints2D_coco = pred_dict['joints2D']  # (bsize, 17, 2) or (num views, 17, 2)
            target_joints2D_coco = target_dict['joints2D']  # (bsize, 17, 2) or (num views, 17, 2)
            joints2D_l2e_batch = np.linalg.norm(pred_joints2D_coco - target_joints2D_coco, axis=-1)  # (bsize, 17) or (num views, 17)
            self.metric_sums['joints2D-L2E'] += np.sum(joints2D_l2e_batch)  # scalar
            self.per_frame_metrics['joints2D-L2E'].append(np.mean(joints2D_l2e_batch, axis=-1))  # (bs,) or (num views,)
            if return_per_frame_metrics:
                per_frame_metrics_return_dict['joints2D-L2E'] = np.mean(joints2D_l2e_batch, axis=-1)

        if 'joints2Dsamples-L2E' in self.metrics_to_track:
            pred_joints2D_coco_samples = pred_dict['joints2Dsamples']  # (bsize, num_samples, 17, 2)
            target_joints2D_coco = np.tile(target_dict['joints2D'][:, None, :, :], (1, pred_joints2D_coco_samples.shape[1], 1, 1))  # (bsize, num_samples, 17, 2)
            if 'joints2D_vis' in target_dict.keys():
                target_joints2d_vis_coco = np.tile(target_dict['joints2D_vis'][:, None, :], (1, pred_joints2D_coco_samples.shape[1], 1))  # (bsize, num_samples, 17)
                pred_joints2D_coco_samples = pred_joints2D_coco_samples[target_joints2d_vis_coco, :]  # (N, 2)
                target_joints2D_coco = target_joints2D_coco[target_joints2d_vis_coco, :]  # (N, 2)
            joints2Dsamples_l2e_batch = np.linalg.norm(pred_joints2D_coco_samples - target_joints2D_coco, axis=-1)  # (N,) or (bsize, num_samples, 17)
            if 'joints2D_vis' in target_dict.keys():
                assert joints2Dsamples_l2e_batch.shape[0] == target_joints2d_vis_coco.sum()
            joints2Dsamples_l2e_batch = joints2Dsamples_l2e_batch.reshape(-1)
            self.metric_sums['joints2Dsamples-L2E'] += np.sum(joints2Dsamples_l2e_batch)  # scalar
            self.metric_sums['num_vis_joints2Dsamples'] += joints2Dsamples_l2e_batch.shape[0]

        if 'silhouette-IOU' in self.metrics_to_track:
            pred_silhouettes = pred_dict['silhouettes']  # (bsize, img_wh, img_wh) or (num views, img_wh, img_wh)
            target_silhouettes = target_dict['silhouettes']  # (bsize, img_wh, img_wh) or (num views, img_wh, img_wh)
            true_positive = np.logical_and(pred_silhouettes, target_silhouettes)
            false_positive = np.logical_and(pred_silhouettes, np.logical_not(target_silhouettes))
            true_negative = np.logical_and(np.logical_not(pred_silhouettes), np.logical_not(target_silhouettes))
            false_negative = np.logical_and(np.logical_not(pred_silhouettes), target_silhouettes)
            num_tp = np.sum(true_positive, axis=(1, 2))  # (bsize,) or (num views,)
            num_fp = np.sum(false_positive, axis=(1, 2))
            num_tn = np.sum(true_negative, axis=(1, 2))
            num_fn = np.sum(false_negative, axis=(1, 2))
            self.metric_sums['num_true_positives'] += np.sum(num_tp)  # scalar
            self.metric_sums['num_false_positives'] += np.sum(num_fp)
            self.metric_sums['num_true_negatives'] += np.sum(num_tn)
            self.metric_sums['num_false_negatives'] += np.sum(num_fn)
            iou_per_frame = num_tp/(num_tp + num_fp + num_fn)
            self.per_frame_metrics['silhouette-IOU'].append(iou_per_frame)  # (bs,) or (num views,)
            if return_per_frame_metrics:
                per_frame_metrics_return_dict['silhouette-IOU'] = iou_per_frame

        if 'silhouettesamples-IOU' in self.metrics_to_track:
            pred_silhouettes_samples = pred_dict['silhouettessamples']  # (bsize, num_samples, img_wh, img_wh)
            target_silhouettes = np.tile(target_dict['silhouettes'][:, None, :, :], (1, pred_silhouettes_samples.shape[1], 1, 1))  # (bsize, num_samples, img_wh, img_wh)
            true_positive = np.logical_and(pred_silhouettes_samples, target_silhouettes)
            false_positive = np.logical_and(pred_silhouettes_samples, np.logical_not(target_silhouettes))
            true_negative = np.logical_and(np.logical_not(pred_silhouettes_samples), np.logical_not(target_silhouettes))
            false_negative = np.logical_and(np.logical_not(pred_silhouettes_samples), target_silhouettes)
            num_tp = np.sum(true_positive, axis=(1, 2))  # (bsize,) or (num views,)
            num_fp = np.sum(false_positive, axis=(1, 2))
            num_tn = np.sum(true_negative, axis=(1, 2))
            num_fn = np.sum(false_negative, axis=(1, 2))
            self.metric_sums['num_samples_true_positives'] += np.sum(num_tp)  # scalar
            self.metric_sums['num_samples_false_positives'] += np.sum(num_fp)
            self.metric_sums['num_samples_true_negatives'] += np.sum(num_tn)
            self.metric_sums['num_samples_false_negatives'] += np.sum(num_fn)

        return transformed_points_return_dict, per_frame_metrics_return_dict

    def compute_final_metrics(self):
        final_metrics = {}
        for metric_type in self.metrics_to_track:
            mult = 1.
            if metric_type == 'silhouette-IOU':
                iou = self.metric_sums['num_true_positives'] / \
                      (self.metric_sums['num_true_positives'] +
                       self.metric_sums['num_false_negatives'] +
                       self.metric_sums['num_false_positives'])
                final_metrics['silhouette-IOU'] = iou
            elif metric_type == 'silhouettesamples-IOU':
                iou = self.metric_sums['num_samples_true_positives'] / \
                      (self.metric_sums['num_samples_true_positives'] +
                       self.metric_sums['num_samples_false_negatives'] +
                       self.metric_sums['num_samples_false_positives'])
                final_metrics['silhouettesamples-IOU'] = iou
            elif metric_type == 'joints2Dsamples-L2E':
                joints2Dsamples_l2e = self.metric_sums['joints2Dsamples-L2E'] / self.metric_sums['num_vis_joints2Dsamples']
                final_metrics[metric_type] = joints2Dsamples_l2e
            else:
                if 'PVE' in metric_type:
                    num_per_sample = 6890
                    mult = 1000.  # mult used to convert 3D metrics from metres to millimetres
                elif 'MPJPE' in metric_type:
                    num_per_sample = 14
                    mult = 1000.
                elif 'joints2D' in metric_type:
                    num_per_sample = 17
                final_metrics[metric_type] = self.metric_sums[metric_type] / (self.total_samples * num_per_sample)

            print(metric_type, '{:.2f}'.format(final_metrics[metric_type] * mult))

        if self.save_per_frame_metrics:
            for metric_type in self.metrics_to_track:
                if 'samples' not in metric_type:
                    per_frame = np.concatenate(self.per_frame_metrics[metric_type], axis=0)
                    np.save(os.path.join(self.save_path, metric_type+'_per_frame.npy'), per_frame)
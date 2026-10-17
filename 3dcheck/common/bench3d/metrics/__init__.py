"""Metrics. Submodules import heavy deps lazily; import what you need:

    from bench3d.metrics.depth import depth_metrics, binned_depth_metrics, error_vs_distance
    from bench3d.metrics.pose import pose_metrics
    from bench3d.metrics.pointcloud import reconstruction_metrics, frustum_mask
    from bench3d.metrics.nvs import psnr, ssim, lpips
    from bench3d.metrics.segmentation import confusion_matrix, iou_from_confusion
    from bench3d.metrics.detection import evaluate_detection, write_custom_splits   # needs nuscenes-devkit
"""

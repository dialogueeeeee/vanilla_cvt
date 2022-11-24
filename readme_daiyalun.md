## command for generate semantic BEV map
### dataset preparation
[Dataset setup](docs/dataset_setup.md).

### generate cvt labels
```bash
# Show labels as they are being processed with the visualization argument
python generate_data.py data=nuscenes data.version=v1.0-mini data.dataset_dir=/home/daiyalun/slam_test/cross_view_transformers-master/datasets/nuscenes data.labels_dir=/home/daiyalun/slam_test/cross_view_transformers-master/datasets/cvt_labels_nuscenes visualization=nuscenes_viz

# Disable visualizations by omitting the "visualization" flag
python generate_data.py data=nuscenes data.version=v1.0-mini data.dataset_dir=/home/daiyalun/slam_test/cross_view_transformers-master/datasets/nuscenes data.labels_dir=/home/daiyalun/slam_test/cross_view_transformers-master/datasets/cvt_labels_nuscenes
```

### show
```bash
python show_daiyalun.py
```
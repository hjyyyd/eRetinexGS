# eRetinexGS: Retinex Modeling for Low-Light Scene Enhancement via Event Streams and 3D Gaussian Splatting


## Data

Place each scene under `datasets/<dataname>/<scene>/`. A scene is a COLMAP reconstruction with
the low-light frames, the (paired) normal-light frames used for evaluation, and per-frame events:

```
datasets/<dataname>/<scene>/
├── sparse/0/            # COLMAP output: cameras.bin, images.bin, points3D.bin
├── images_lowImg/       # low-light input frames
├── images_norImg/       # normal-light frames (ground truth, for evaluation)
└── event_npy/           # per-frame event maps (.npy, one per frame)
```

The released configuration uses the low-light LLFF dataset
`nerf_llff_data_evllgs` (8 scenes: `fern, flower, fortress,
horns, leaves, orchids, room, trex`).




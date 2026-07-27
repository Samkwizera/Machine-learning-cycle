# Data

The dataset is the TensorFlow Flowers set: ~3,600 photos across five classes
(daisy, dandelion, roses, sunflowers, tulips).

The images are not committed — they are ~220 MB and are reproducible in one
command:

```bash
python src/data_acquisition.py
```

That downloads the archive and fills these folders with a 70/15/15 split:

```
data/
├── train/<class>/    training images
├── val/<class>/      validation images, used for early stopping
├── test/<class>/     held-out test images, never seen during training
├── uploads/<class>/  images added through the app's Upload Data page
├── app.db            SQLite log of uploads and retraining runs
└── viz_cache.json    precomputed aggregates so the visualizations work on the
                      hosted app, which has no dataset
```

The split is done once with a fixed seed, so re-running the script gives the
same files in the same folders.

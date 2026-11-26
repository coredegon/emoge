# data

Datasets come in many layouts; `emoge.data` reads them into one. A sample is an
image as RGB bytes, the fields of its annotation, and its label.

```python
from pathlib import Path

from PIL import Image
from emoge import color, data

# A tiny dataset to read: a folder per class.
for path, rgb in [
    ("calm/1.png", (60, 90, 200)),
    ("calm/2.png", (70, 160, 90)),
    ("fear/1.png", (20, 20, 30)),
]:
    Path("photos", path).parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (40, 30), rgb).save(Path("photos", path))
```

Everything below is at `data.`.

## A folder per class

```python
images = data.folders("photos", size=16)
print(images, images.classes)  # Dataset(3 images, 2 classes) ['calm', 'fear']
sample = images[2]
print(sample["image"].shape, sample["label"])  # (16, 16, 3) 1
```

A sample is a dict: `"image"`, the RGB bytes; `"path"`, its file; `"label"`, an
index into `classes`; and the other fields of its annotation.

`size` makes every image `size` x `size`. The shorter side shrinks, every new
pixel the mean of those it covers, so colors keep their shares; then the middle
is cut out. Without `size`, images keep theirs.

## A table of images

`table(path, image=..., label=...)` reads a CSV or JSON file that lists images,
one a row; its other columns become fields of the samples. Paths are relative to
the table.

```python
Path("photos/train.csv").write_text(
    "path,emotion,valence\ncalm/1.png,calm,0.7\nfear/1.png,fear,0.1\n"
)
listed = data.table("photos/train.csv", image="path", label="emotion")
print(listed[1]["label"], listed[1]["valence"])  # 1 0.1
```

Values of a CSV stay text; JSON keeps numbers and lists as they are. Rows from
anywhere else make a dataset too: `data.Dataset(rows, root, label=...)`.

## Features of a dataset

`features(dataset, function)` gives `function` of every image, stacked into one
array, loading the images in threads; `labels` gives every label without
loading them:

```python
palettes = data.features(images, color.COLIBRI.palette)
print(palettes.shape, images.labels.tolist())  # (3, 86) [0, 0, 1]
```

`each(dataset, function)` gives the same results one by one, in order, for
results too large to keep all at once, such as graphs:

```python
for palette in data.each(images, color.COLIBRI.palette):
    print(palette.shape)
# (86,)
# (86,)
# (86,)
```

A dataset is a sequence of samples, so `torch.utils.data.DataLoader(images,
batch_size=32)` takes it as it is.

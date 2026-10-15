"""Optional fast CIFAR-10 mirror preparation; requires pyarrow==19.0.1.
Source: https://huggingface.co/datasets/uoft-cs/cifar10
No labels or split membership are changed.
"""
import hashlib
from io import BytesIO
from pathlib import Path
import urllib.request
import numpy as np
from PIL import Image
import pyarrow.parquet as pq

HASHES = {'train': '8428b53a88a11ac374111006708df51469e315a22ac6d66470afd9c78d2ae883',
          'test': '841389e6f2d64f28bf17310e430aebac20ec3ba611a3c5e231dc93c645ce84de'}
Path('data').mkdir(exist_ok=True)
for split in ('train', 'test'):
    path = Path(f'data/{split}.parquet')
    if not path.exists():
        urllib.request.urlretrieve(f'https://huggingface.co/datasets/uoft-cs/cifar10/resolve/main/plain_text/{split}-00000-of-00001.parquet?download=true', path)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == HASHES[split], 'Mirror file hash mismatch'
    table = pq.read_table(path).to_pydict()
    images = np.stack([np.asarray(Image.open(BytesIO(row['bytes'])).convert('RGB')) for row in table['img']])
    labels = np.asarray(table['label'], dtype=np.int64)
    np.savez(f'data/{split}.npz', images=images, labels=labels)
    print(split, images.shape, labels.shape, flush=True)

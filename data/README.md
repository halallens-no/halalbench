# HalalBench Dataset

## Overview

The HalalBench dataset contains **1,043 food packaging images** with **36,438 text annotations** across **14 languages**, annotated in COCO format with bounding boxes and transcriptions for ingredient text regions.

## Download

The dataset will be available for download at:

```
https://github.com/halallens-no/halalbench/releases
```

After downloading, extract into this directory:

```bash
cd data/
tar -xzf halalbench-v1.0.tar.gz
```

This will produce:

```
data/
  annotations.json     # COCO-format ground truth
  images/              # 1,043 food packaging images
    img_0001.jpg
    img_0002.jpg
    ...
```

## COCO Annotation Format

The annotations follow the standard COCO format with additional fields for OCR:

```json
{
  "images": [
    {
      "id": 1,
      "file_name": "img_0001.jpg",
      "width": 1920,
      "height": 1080,
      "language": "en"
    }
  ],
  "annotations": [
    {
      "id": 1,
      "image_id": 1,
      "bbox": [x, y, width, height],
      "category_id": 1,
      "attributes": {
        "text": "sodium benzoate",
        "confidence": 1.0
      }
    }
  ],
  "categories": [
    {
      "id": 1,
      "name": "ingredient_text"
    }
  ]
}
```

### Key Fields

| Field | Description |
|-------|-------------|
| `images[].language` | ISO 639-1 language code of the primary text |
| `annotations[].bbox` | Bounding box in COCO format `[x, y, w, h]` |
| `annotations[].attributes.text` | Ground-truth transcription of the text region |

## Language Distribution

| Language | Code | Images | Annotations |
|----------|------|--------|-------------|
| Arabic | ar | 72 | 2,518 |
| Danish | da | 68 | 2,380 |
| Dutch | nl | 71 | 2,485 |
| English | en | 112 | 3,920 |
| French | fr | 89 | 3,115 |
| German | de | 85 | 2,975 |
| Indonesian | id | 62 | 2,170 |
| Japanese | ja | 78 | 2,730 |
| Korean | ko | 74 | 2,590 |
| Malay | ms | 59 | 2,065 |
| Norwegian | no | 65 | 2,275 |
| Swedish | sv | 70 | 2,450 |
| Thai | th | 69 | 2,415 |
| Turkish | tr | 69 | 2,350 |
| **Total** | | **1,043** | **36,438** |

## Annotation Guidelines

Each image was annotated by at least two annotators with the following protocol:

1. **Bounding boxes** tightly enclose each ingredient text region
2. **Transcriptions** preserve the original text as printed (including accents, special characters)
3. **Language labels** reflect the primary language of the ingredient list on each image
4. Inter-annotator disagreements resolved by a third annotator

## License

The HalalBench dataset is licensed under **Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0)**.

You are free to share and adapt the dataset, provided you give appropriate credit and distribute any derivative works under the same license.

Full terms: https://creativecommons.org/licenses/by-sa/4.0/

## Citation

```bibtex
@article{halalbench2026,
  title     = {HalalBench: A Multilingual OCR Benchmark for Food Packaging Ingredient Extraction},
  author    = {HalalLens Research},
  journal   = {arXiv preprint arXiv:XXXX.XXXXX},
  year      = {2026}
}
```

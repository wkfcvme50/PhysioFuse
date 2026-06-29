# Example Data

This folder contains two renamed paired examples for checking the expected input layout.

```text
example_data/
  hbs.csv
  conjunctiva/
    sample_001.jpg
    sample_002.jpg
    conjunctiva_masks/
      sample_001.png
      sample_002.png
  nail/
    sample_001.jpg
    sample_002.jpg
    nail_masks/
      sample_001.png
      sample_002.png
```

`hbs.csv` uses the same minimal columns as the private training label file:

```text
name,hb
```

The original clinical identifiers are not included. The included mask files are renamed and metadata-stripped binary ROI masks for these two examples only. The data loader also supports missing masks by falling back to a full-image mask for format testing. The original clinical dataset, full labels, full ROI mask set, sample identifiers, checkpoints, and sample-level outputs are not released.

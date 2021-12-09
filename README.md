# FloodNet_public
Semantic segmentation experiment on the FloodNet Challenge Dataset

Setup instructions:
- Download the FloodNet Dataset at https://github.com/BinaLab/FloodNet-Challenge-EARTHVISION2021
- Download latest versions of the required ML and image processing packages for your Python environment (I recommend using Anaconda for this step to make environment management easier)
  - Python 3
  - numpy
  - Pytorch with GPU acceleration (follow instructions on official website)
  - Pillow
  - matplotlib
  - ml_collections
- Organize the folder as such

```
dataset
|- Train
|- Validation
|- Test
|- class_mapping.csv
<all .py files here>
```

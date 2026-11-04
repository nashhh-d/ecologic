# ECOLOGIC — Backend / AI-ML work package

This package contains the backend-owned modules for the ECOLOGIC Smart Waste-to-Microbe Converter.

## Modules
- `prepare_dataset.py`: organizes the downloaded image dataset into `ImageFolder` directories; creates a per-class train/validation split if the source has no split folders.
- `train.py`: fine-tunes MobileNetV3-Small and saves the best validation macro-F1 checkpoint.
- `ecologic/inference.py`: loads the trained checkpoint and predicts a category from a PIL image.
- `ecologic/recommender.py`: transparent rule-based candidate treatment recommendations.
- `ecologic/service.py`: joins prediction and recommendation into one function for the Streamlit frontend.

## Dataset
Use the selected Kaggle Waste Segregation Image Dataset:
https://www.kaggle.com/dsv/4235079

Read its dataset card and license before use. The listing previously identified CC BY-NC-SA 4.0 and non-commercial restrictions. Preserve attribution and use only within the permitted terms.

## Setup (Windows)
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Prepare data
Download and extract the dataset. Then run (replace the example path):
```bash
python prepare_dataset.py --source "C:\path\to\extracted_dataset" --output data/organic_waste
```
If you have rerun the preparation script and want to replace prior copied images:
```bash
python prepare_dataset.py --source "C:\path\to\extracted_dataset" --output data/organic_waste --clear
```
Inspect the output counts and class names. If it says no recognized classes, send the extracted folder tree so the class mapping can be adjusted. Do not assume every version of a dataset has the same structure.

## Train
```bash
python train.py --data-dir data/organic_waste --epochs 8 --batch-size 32
```
Pretrained weights may need internet access on the first run. For a no-download experiment, use `--no-pretrained`, though transfer learning is generally a better starting point.

The checkpoint is saved at:
`outputs/ecologic_mobilenetv3.pth`

## Use inference in Python
```python
from PIL import Image
from ecologic.service import analyze_image

image = Image.open("sample_waste.jpg")
result = analyze_image(image)
print(result["prediction"])
print(result["recommendation"])
```

The Streamlit frontend can import `analyze_image` from `ecologic.service`.

## Scientific limitations
- Image classification identifies visual categories; it does not directly measure pH, moisture, C:N ratio, volatile solids or microbial activity.
- The treatment engine gives candidate pathways based on broad material categories, not validated process-control instructions.
- A low-confidence prediction is routed to manual review.
- Report actual measured validation/test results only. Do not invent accuracy or environmental savings.

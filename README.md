# 🧠 Object Detection Using TensorFlow Object Detection API

This project demonstrates how to build and train a **custom object detection model** using the TensorFlow Object Detection API. It provides a step-by-step guide for dataset preparation, model training, evaluation, and inference.

![TensorFlow Object Detection](https://www.tensorflow.org/images/object_detection/architecture.svg)

## 📌 Overview

- 🔍 Custom object detection with TensorFlow 2.x
- 🗂️ TFRecord generation for training and testing
- 🧪 Model training and evaluation
- 📦 Exporting and running inference on test images
- 📒 Jupyter Notebook for easy experimentation

## 📁 Project Structure

```
├── object_detection_USINGTFOD.ipynb     # Main Jupyter Notebook
├── /annotations/                        # Label map & TFRecord files
├── /images/                             # Dataset (train/test images)
├── /models/                             # Saved models & pipeline configs
├── /pre-trained-models/                 # Pretrained models from Model Zoo
└── README.md                            # Project documentation
```

## ⚙️ Requirements

- Python 3.7+
- TensorFlow 2.x
- TensorFlow Object Detection API
- OpenCV
- LabelImg (for annotations)

Install all dependencies:

```bash
pip install tensorflow opencv-python matplotlib pandas
```

Set up the Object Detection API:

```bash
git clone https://github.com/tensorflow/models.git
cd models/research/
protoc object_detection/protos/*.proto --python_out=.
cp object_detection/packages/tf2/setup.py .
python -m pip install .
```

## 📷 Dataset Preparation

1. Annotate your images using [LabelImg](https://github.com/tzutalin/labelImg).
2. Create `label_map.pbtxt` inside the `/annotations` folder.
3. Convert `.xml` files to `.tfrecord` format for both training and testing using a conversion script.

## 🏁 How to Use

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/object-detection-tfod.git
cd object-detection-tfod
```

### 2. Open the Notebook

Launch the Jupyter notebook and run through all the cells in `object_detection_USINGTFOD.ipynb`.

```bash
jupyter notebook
```

### 3. Train the Model

- Configure the `pipeline.config` file with paths to TFRecords, label maps, and pretrained model.
- Use the notebook to initiate model training or run the command-line script.

### 4. Export the Trained Model

```bash
python exporter_main_v2.py \
    --input_type image_tensor \
    --pipeline_config_path models/my_model/pipeline.config \
    --trained_checkpoint_dir models/my_model \
    --output_directory exported-models/my_model
```

### 5. Run Inference

Load the exported model and test it on custom images.

## 📊 Results

- Model: e.g., SSD MobileNet V2
- Training images: 500
- Evaluation mAP: _Add your results here_

## 📎 Tips

- Start with a pretrained model for better accuracy and faster convergence.
- Use balanced datasets with varied object instances.
- Fine-tune learning rates and augmentation if performance is poor.

## 🤝 Acknowledgements

- [TensorFlow Models GitHub](https://github.com/tensorflow/models)
- [LabelImg](https://github.com/tzutalin/labelImg)

---

### ⭐ Star this repo if you find it helpful!

> Feel free to raise issues or submit pull requests if you find bugs or want to contribute improvements.

# Alzheimer's Disease Classification Model

This project contains a deep learning model for classifying Alzheimer's disease stages from brain MRI scans.

## Dataset Requirements

The model requires the Alzheimer's dataset with the following structure:

```
Alzheimer_s Dataset/
└── train/
    ├── MildDemented/
    │   └── (MRI images...)
    ├── ModerateDemented/
    │   └── (MRI images...)
    ├── NonDemented/
    │   └── (MRI images...)
    └── VeryMildDemented/
        └── (MRI images...)
```

## Getting the Dataset

1. Download the Alzheimer's dataset from Kaggle:
   https://www.kaggle.com/datasets/tourist55/alzheimers-dataset-4-class-of-images

2. Extract the zip file

3. Place the extracted folders in the correct structure as shown above

## Running the Model

Run the model with:

```
python new.py
```

The script will:
1. Check if the dataset is available
2. Allow you to specify a different dataset location if needed
3. Train the model on the dataset
4. Evaluate the model performance
5. Save the trained model

## Model Features

- Uses EfficientNetB0 as the base model
- Implements data augmentation to reduce overfitting
- Includes regularization techniques
- Provides visualizations for model performance
- Generates interpretability visualizations with Grad-CAM 
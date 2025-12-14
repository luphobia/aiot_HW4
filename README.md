# Pet Emotion Classifier

This project is a pet emotion classifier that can predict the emotion of a cat or dog from an image. It uses a deep learning model trained on a dataset of pet facial expressions. The application is built with Streamlit and uses a YOLOv8 model for object detection.

## Features

-   Web interface for uploading images or selecting sample images.
-   Emotion classification for cats and dogs.
-   Optional object detection to crop the pet from the image.
-   Displays the predicted emotion with a confidence score.

## Project Structure

```
├── data/                    # Dataset directory
├── models/                  # Trained models
├── samples/                 # Sample images
├── detector.py              # Pet detection and cropping
├── prepare_data.py          # Data preparation script
├── preprocess_split_species.py # script for splitting the dataset by species
├── requirements.txt         # Project dependencies
├── streamlit_app.py         # Main application file
├── train.py                 # Model training script
└── yolov8n.pt               # YOLOv8 model weights
```

## Installation

1.  Clone the repository:
    ```
    git clone <repository-url>
    ```
2.  Install the required dependencies:
    ```
    pip install -r requirements.txt
    ```

## Usage

### 1. Prepare the Data

To train the model, you first need to download and prepare the dataset. This can be done by running the `prepare_data.py` script:

```
python prepare_data.py
```

This will download the dataset from Kaggle and organize it into `train` and `val` directories.

### 2. Train the Model

To train the emotion classification model, run the `train.py` script:

```
python train.py
```

This will train the model and save the best-performing weights to the `models` directory.

### 3. Run the Application

To run the Streamlit application, use the following command:

```
streamlit run streamlit_app.py
```

This will open the application in your web browser, where you can upload an image or select a sample to classify.

## File Descriptions

-   **`streamlit_app.py`**: The main application file that creates the web interface using Streamlit.
-   **`detector.py`**: Contains the `PetDetector` class, which uses a YOLOv8 model to detect and crop pets from an image.
-   **`train.py`**: The script for training the emotion classification model.
-   **`prepare_data.py`**: A script for downloading and preparing the dataset.
-   **`preprocess_split_species.py`**: A script for splitting the dataset by species (cat or dog).
-   **`requirements.txt`**: A list of the Python dependencies required to run the project.
-   **`yolov8n.pt`**: The model weights for the YOLOv8 object detection model.
-   **`data/`**: The directory where the dataset is stored.
-   **`models/`**: The directory where the trained models are saved.
-   **`samples/`**: A directory containing sample images for testing the application.

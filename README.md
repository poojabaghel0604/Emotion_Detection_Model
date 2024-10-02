---

# Emotion Detection Using CNN and FER-2013 Dataset 😄😢😡

This project is focused on detecting emotions from images and live video streams using **Convolutional Neural Networks (CNNs)**. It utilizes the **FER-2013 dataset** and includes a **real-time emotion detection system** deployed with **Gradio** and **OpenCV**. The model is designed to recognize 7 distinct emotions and is optimized using advanced CNN architectures like **ResNet50v2** and **VGG16**.

## 📚 Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Screenshots](#screenshots)
- [Technologies Used](#technologies-used)
- [Results](#results)
- [Future Enhancements](#future-enhancements)
- [Contributing](#contributing)
- [License](#license)

## 🌟 Overview
Emotion detection is a crucial component in human-computer interaction, enabling machines to interpret and respond to human emotional states. This project focuses on building a **real-time emotion detection system** using a deep learning model trained on the **FER-2013 dataset**.

The project includes:
- **Data Preprocessing and Augmentation**: To address class imbalance and improve model performance.
- **Custom CNN Models**: Including architectures like **VGG16** and **ResNet50v2** to optimize classification accuracy.
- **Real-time Emotion Detection**: Integrated with **OpenCV** for video feed processing and **Gradio** for a user-friendly interface.

## ✨ Features
- **Real-time emotion detection** in live video streams.
- **Emotion classification** for 7 emotion categories: Anger, Disgust, Fear, Happy, Sad, Surprise, and Neutral.
- **Gradio-based interface** for easy interaction and testing of the emotion detection model.
- **Custom CNN architectures** like **VGG16** and **ResNet50v2** for optimized performance.
- **FER-2013 dataset** used for training and evaluation.

## 📊 Dataset
The **FER-2013** dataset is a popular dataset used for facial emotion recognition. It consists of:
- 35,887 grayscale, 48x48 pixel face images categorized into 7 different emotions:
  - Anger
  - Disgust
  - Fear
  - Happy
  - Sad
  - Surprise
  - Neutral
- Data augmentation techniques were employed to balance the class distribution and improve the model’s robustness.

## 🏛️ Model Architecture
The final model is built using **ResNet50v2**, a residual learning framework designed to ease the training of very deep networks. Additionally, **VGG16** and custom CNN models were explored to find the optimal architecture.

Key steps taken to improve the model:
- **Class Weights**: Addressed class imbalance by assigning weights to different emotion classes.
- **Image Augmentation**: Improved generalization by applying transformations like rotation, zoom, flip, and more.
- **Transfer Learning**: Leveraged pre-trained models (VGG16 and ResNet50v2) to enhance model accuracy.

The final model achieved a **66% overall accuracy** in classifying 7 emotion labels.

## ⚙️ Installation

To run the project locally, follow the steps below:

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/emotion-detection-cnn.git
    ```
2. Navigate to the project directory:
    ```bash
    cd emotion-detection-cnn
    ```
3. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```
4. Run the Gradio interface:
    ```bash
    python app.py
    ```

## 🚀 Usage
### Real-time Emotion Detection:
1. Run the Gradio app.
2. Upload an image to see the predicted emotion.
3. For live video emotion detection, use OpenCV to open the webcam feed.

### Example:
- Upload a photo, and the model will predict the emotion as `Happy`, `Sad`, `Angry`, etc.
- In live video mode, emotions are detected and displayed dynamically on the screen.

## 📸 Screenshots

### Gradio Interface
![Emotion Detection Gradio Interface](https://github.com/poojabaghel0604/Emotion_Detection_Model/blob/main/Screenshot%20(71).png)

### Real-Time Emotion Detection
![Real-Time Emotion Detection](https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQVkOGYOBkHRxAl0wCFSiHn3EFGX4CqwMv6ng&s)

## 🛠️ Technologies Used
- **Python**: Core language used for development.
- **TensorFlow** & **Keras**: Used for building and training CNN models.
- **ResNet50v2 & VGG16**: Pre-trained CNN models used for transfer learning.
- **OpenCV**: For real-time video processing.
- **Gradio**: For building the user-friendly web interface.
- **Pandas, Numpy, Matplotlib**: For data handling and visualization.

## 🏆 Results
The final model using **ResNet50v2** achieved:
- **66% accuracy** across 7 emotion labels.
- Metrics such as **precision, recall, and F1-scores** were evaluated for each emotion category to analyze performance.

| Emotion | Precision | Recall | F1-Score |
|---------|-----------|--------|----------|
| Angry   | 0.54      | 0.57   | 0.55     |
| Disgust | 0.57      | 0.61   | 0.59     |
| Fear    | 0.43      | 0.37   | 0.40     |
| Happy   | 0.89      | 0.79   | 0.84     |
| Sad     | 0.53      | 0.63   | 0.58     |
| Surprise| 0.55      | 0.39   | 0.46     |
| Neutral | 0.56      | 0.84   | 0.67     |

## 🔮 Future Enhancements
Some potential improvements include:
- **Improve model accuracy** using more advanced architectures and larger datasets.
- **Multi-emotion detection**: Allow the system to detect multiple emotions simultaneously.
- **Emotion detection in groups**: Enable the system to detect emotions in group photos.
- **Deployment**: Deploy the model as a cloud-based service for wider access.

## 🤝 Contributing
Contributions are welcome! Please follow these steps:

1. Fork the repository.
2. Create a new feature branch (`git checkout -b feature/NewFeature`).
3. Commit your changes (`git commit -m 'Add new feature'`).
4. Push to the branch (`git push origin feature/NewFeature`).
5. Open a pull request.

## 📜 License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from ultralytics import YOLO
from tqdm import tqdm
import pandas as pd
from PIL import Image

# Define constants
EMOTIONS = ['Happy', 'Sad', 'Angry', 'Surprise', 'Neutral', 'Fear', 'Disgust']
NUM_EMOTIONS = len(EMOTIONS)
IMAGE_SIZE = 48  # Input size for the emotion classifier
SEQUENCE_LENGTH = 10  # Number of frames to consider for temporal analysis
BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 0.001

# Load YOLOv8 for face detection
def load_face_detector():
    # Download YOLOv8n-face model or use a pre-downloaded one
    try:
        model = YOLO('yolov8n-face.pt')
    except:
        print("Downloading YOLOv8n-face model...")
        model = YOLO('yolov8n-face.pt')
    return model

# Face Detection Function
def detect_faces(image, model, conf_threshold=0.5):
    results = model(image)
    faces = []
    
    if results[0].boxes.data.shape[0] > 0:
        for box in results[0].boxes.data:
            if box[4] >= conf_threshold:  # Check confidence
                x1, y1, x2, y2 = map(int, box[:4])
                faces.append((x1, y1, x2, y2))
    
    return faces

# Function to preprocess faces
def preprocess_face(image, face_coords, target_size=(48, 48)):
    x1, y1, x2, y2 = face_coords
    face = image[y1:y2, x1:x2]
    
    # Convert to grayscale
    if len(face.shape) == 3 and face.shape[2] == 3:
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    
    # Resize
    face = cv2.resize(face, target_size)
    
    return face

# Custom Dataset for FER-2013
class FER2013Dataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        pixels, emotion = self.data[idx]
        
        # Convert to PIL Image
        image = Image.fromarray(pixels.astype('uint8'))
        
        if self.transform:
            image = self.transform(image)
        
        return image, emotion

# Custom Dataset for sequence data (for LSTM)
class EmotionSequenceDataset(Dataset):
    def __init__(self, data, sequence_length, transform=None):
        self.data = data
        self.sequence_length = sequence_length
        self.transform = transform
        
        # Create sequences
        self.sequences = []
        for i in range(len(data) - sequence_length + 1):
            self.sequences.append((data[i:i+sequence_length]))
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        images = []
        labels = []
        
        for pixels, emotion in sequence:
            image = Image.fromarray(pixels.astype('uint8'))
            
            if self.transform:
                image = self.transform(image)
            
            images.append(image)
            labels.append(emotion)
        
        # Use the last emotion in sequence as the target
        return torch.stack(images), labels[-1]

# CNN Model for Emotion Recognition
class EmotionCNN(nn.Module):
    def __init__(self, num_emotions):
        super(EmotionCNN, self).__init__()
        
        # CNN Layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        
        # Pooling and Dropout
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout2d(0.5)
        
        # Calculate the size after convolutions and pooling
        # Starting with 48x48, after 4 pooling layers: 3x3x256
        self.fc_input_size = 3 * 3 * 256
        
        # Fully Connected Layers
        self.fc1 = nn.Linear(self.fc_input_size, 512)
        self.bn_fc = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, num_emotions)
        
        # Activation
        self.relu = nn.ReLU()
    
    def forward(self, x):
        # Convolutional layers
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        x = self.relu(self.bn4(self.conv4(x)))
        x = self.pool(x)
        
        # Flatten
        x = x.view(-1, self.fc_input_size)
        
        # Fully connected layers
        x = self.relu(self.bn_fc(self.fc1(x)))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x
    
    def extract_features(self, x):
        # This method returns features before the final classification layer
        # Convolutional layers
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        x = self.relu(self.bn4(self.conv4(x)))
        x = self.pool(x)
        
        # Flatten
        x = x.view(-1, self.fc_input_size)
        
        # First FC layer
        x = self.relu(self.bn_fc(self.fc1(x)))
        
        return x

# CNN-LSTM Model for Emotion Recognition
class EmotionCNNLSTM(nn.Module):
    def __init__(self, cnn_model, num_emotions, sequence_length, hidden_size=128):
        super(EmotionCNNLSTM, self).__init__()
        
        # CNN Feature Extractor (pre-trained)
        self.cnn = cnn_model
        self.feature_size = 512  # Size of features from CNN
        
        # LSTM for temporal analysis
        self.lstm = nn.LSTM(
            input_size=self.feature_size,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=0.5
        )
        
        # Final classification layer
        self.fc = nn.Linear(hidden_size, num_emotions)
        
        # Sequence length
        self.sequence_length = sequence_length
    
    def forward(self, x_sequence):
        batch_size = x_sequence.size(0)
        
        # Reshape for CNN processing
        x_sequence = x_sequence.view(-1, 1, IMAGE_SIZE, IMAGE_SIZE)
        
        # Extract features using CNN
        features = self.cnn.extract_features(x_sequence)
        
        # Reshape for LSTM processing
        features = features.view(batch_size, self.sequence_length, -1)
        
        # LSTM processing
        lstm_out, _ = self.lstm(features)
        
        # Take the output from the last time step
        lstm_out = lstm_out[:, -1, :]
        
        # Final classification
        output = self.fc(lstm_out)
        
        return output

# Function to load data from folders instead of CSV
def load_from_folders(train_dir, test_dir):
    train_data = []
    test_data = []
    
    # Define emotion mapping (folder names to emotion indices)
    emotion_map = {
        'angry': 2, 'disgust': 6, 'fear': 5, 'happy': 0, 
        'neutral': 4, 'sad': 1, 'surprise': 3
    }
    
    # Load training data
    for emotion_folder in os.listdir(train_dir):
        emotion_path = os.path.join(train_dir, emotion_folder)
        if os.path.isdir(emotion_path):
            emotion_idx = emotion_map.get(emotion_folder.lower(), 0)
            for img_file in os.listdir(emotion_path):
                if img_file.endswith(('.jpg', '.png', '.jpeg')):
                    img_path = os.path.join(emotion_path, img_file)
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    if img is not None:
                        # Resize if necessary
                        img = cv2.resize(img, (48, 48))
                        train_data.append((img, emotion_idx))
    
    # Load test data
    for emotion_folder in os.listdir(test_dir):
        emotion_path = os.path.join(test_dir, emotion_folder)
        if os.path.isdir(emotion_path):
            emotion_idx = emotion_map.get(emotion_folder.lower(), 0)
            for img_file in os.listdir(emotion_path):
                if img_file.endswith(('.jpg', '.png', '.jpeg')):
                    img_path = os.path.join(emotion_path, img_file)
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    if img is not None:
                        # Resize if necessary
                        img = cv2.resize(img, (48, 48))
                        test_data.append((img, emotion_idx))
    
    print(f"Loaded {len(train_data)} training samples and {len(test_data)} test samples")
    return train_data, test_data

# Function to train the CNN model
def train_cnn(model, train_loader, val_loader, criterion, optimizer, num_epochs, device):
    best_val_acc = 0.0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for inputs, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} - Training'):
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Backward + optimize
            loss.backward()
            optimizer.step()
            
            # Statistics
            train_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        train_loss = train_loss / len(train_loader.dataset)
        train_acc = 100.0 * train_correct / train_total
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} - Validation'):
                inputs, labels = inputs.to(device), labels.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_loss = val_loss / len(val_loader.dataset)
        val_acc = 100.0 * val_correct / val_total
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f'Epoch {epoch+1}/{num_epochs}: '
              f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, '
              f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_emotion_cnn.pth')
            print(f'Model saved with validation accuracy: {val_acc:.2f}%')
    
    return history

# Function to train the CNN-LSTM model
def train_cnn_lstm(model, train_loader, val_loader, criterion, optimizer, num_epochs, device):
    best_val_acc = 0.0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for inputs, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} - Training'):
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Backward + optimize
            loss.backward()
            optimizer.step()
            
            # Statistics
            train_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        train_loss = train_loss / len(train_loader.dataset)
        train_acc = 100.0 * train_correct / train_total
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} - Validation'):
                inputs, labels = inputs.to(device), labels.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_loss = val_loss / len(val_loader.dataset)
        val_acc = 100.0 * val_correct / val_total
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f'Epoch {epoch+1}/{num_epochs}: '
              f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, '
              f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_emotion_cnn_lstm.pth')
            print(f'Model saved with validation accuracy: {val_acc:.2f}%')
    
    return history

# Function to evaluate the model
def evaluate_model(model, test_loader, device, emotions):
    y_true = []
    y_pred = []
    
    model.eval()
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc='Evaluating'):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    conf_matrix = confusion_matrix(y_true, y_pred)
    
    # Print metrics
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=emotions, yticklabels=emotions)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')
    plt.close()
    
    return accuracy, precision, recall, f1, conf_matrix

# Real-time emotion detection function
def real_time_emotion_detection(face_detector, emotion_model, device, sequence_length=SEQUENCE_LENGTH):
    cap = cv2.VideoCapture(0)
    frame_queue = []
    
    emotion_model.eval()
    
    # Set video frame width and height
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    # Define data transform for preprocessing frames
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    print("Press 'q' to quit")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect faces
        faces = detect_faces(frame, face_detector)
        
        for (x1, y1, x2, y2) in faces:
            # Draw rectangle around the face
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Preprocess face
            face = preprocess_face(frame, (x1, y1, x2, y2), (IMAGE_SIZE, IMAGE_SIZE))
            
            # Convert to tensor and add to queue
            face_tensor = transform(Image.fromarray(face))
            
            if len(frame_queue) >= sequence_length:
                frame_queue.pop(0)
            frame_queue.append(face_tensor)
            
            # Process sequence if we have enough frames
            if len(frame_queue) == sequence_length:
                with torch.no_grad():
                    # For CNN-LSTM model
                    input_sequence = torch.stack(frame_queue).unsqueeze(0).to(device)
                    output = emotion_model(input_sequence)
                    _, predicted = torch.max(output, 1)
                    emotion = EMOTIONS[predicted.item()]
                
                # Display emotion
                cv2.putText(frame, emotion, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        # Display the frame
        cv2.imshow('Emotion Detection', frame)
        
        # Break loop on 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

# Main function to run the complete pipeline
def main():
    print("Emotion Detection System using YOLOv8 and CNN-LSTM")
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Paths to training and test directories
    train_dir = 'D:/Khushi Kathuria/OpenCV examples/OpenCV/SENSOR/train'
    test_dir = 'D:/Khushi Kathuria/OpenCV examples/OpenCV/SENSOR/test'
    
    # Check if directories exist
    if not (os.path.exists(train_dir) and os.path.exists(test_dir)):
        print("Error: Training or test directory not found.")
        return
    
    # Load data from folders
    print("Loading dataset from folders...")
    train_data, test_data = load_from_folders(train_dir, test_dir)
    
    # Split training data to create validation set
    train_data, val_data = train_test_split(train_data, test_size=0.2, random_state=42)
    
    print(f"Train size: {len(train_data)}, Validation size: {len(val_data)}, Test size: {len(test_data)}")
    
    # Data augmentation and normalization for training
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    # Just normalization for validation/test
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    # Create datasets
    train_dataset = FER2013Dataset(train_data, transform=train_transform)
    val_dataset = FER2013Dataset(val_data, transform=val_transform)
    test_dataset = FER2013Dataset(test_data, transform=val_transform)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    # 1. Train and evaluate the CNN model
    print("\n--- Training CNN Model ---")
    cnn_model = EmotionCNN(NUM_EMOTIONS).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(cnn_model.parameters(), lr=LEARNING_RATE)
    
    # Check if pre-trained model exists
    if os.path.exists('best_emotion_cnn.pth'):
        print("Loading pre-trained CNN model...")
        cnn_model.load_state_dict(torch.load('best_emotion_cnn.pth', map_location=device))
    else:
        print("Training CNN model...")
        cnn_history = train_cnn(
            cnn_model, train_loader, val_loader, criterion, optimizer, EPOCHS, device
        )
        
        # Plot training history
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(cnn_history['train_loss'], label='Train Loss')
        plt.plot(cnn_history['val_loss'], label='Validation Loss')
        plt.title('CNN Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(cnn_history['train_acc'], label='Train Accuracy')
        plt.plot(cnn_history['val_acc'], label='Validation Accuracy')
        plt.title('CNN Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('cnn_training_history.png')
        plt.close()
    
    # Evaluate CNN model
    print("\n--- Evaluating CNN Model ---")
    cnn_accuracy, cnn_precision, cnn_recall, cnn_f1, _ = evaluate_model(
        cnn_model, test_loader, device, EMOTIONS
    )
    
    # 2. Create sequence datasets for LSTM
    train_seq_dataset = EmotionSequenceDataset(train_data, SEQUENCE_LENGTH, transform=train_transform)
    val_seq_dataset = EmotionSequenceDataset(val_data, SEQUENCE_LENGTH, transform=val_transform)
    test_seq_dataset = EmotionSequenceDataset(test_data, SEQUENCE_LENGTH, transform=val_transform)
    
    # Create sequence data loaders
    train_seq_loader = DataLoader(train_seq_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_seq_loader = DataLoader(val_seq_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    test_seq_loader = DataLoader(test_seq_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    # 3. Train and evaluate the CNN-LSTM model
    print("\n--- Training CNN-LSTM Model ---")
    # Set CNN in evaluation mode (for feature extraction)
    cnn_model.eval()
    
    # Create CNN-LSTM model
    cnn_lstm_model = EmotionCNNLSTM(cnn_model, NUM_EMOTIONS, SEQUENCE_LENGTH).to(device)
    
    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(cnn_lstm_model.parameters(), lr=LEARNING_RATE)
    
    # Check if pre-trained model exists
    if os.path.exists('best_emotion_cnn_lstm.pth'):
        print("Loading pre-trained CNN-LSTM model...")
        cnn_lstm_model.load_state_dict(torch.load('best_emotion_cnn_lstm.pth', map_location=device))
    else:
        print("Training CNN-LSTM model...")
        cnn_lstm_history = train_cnn_lstm(
            cnn_lstm_model, train_seq_loader, val_seq_loader, criterion, optimizer, EPOCHS, device
        )
        
        # Plot training history
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(cnn_lstm_history['train_loss'], label='Train Loss')
        plt.plot(cnn_lstm_history['val_loss'], label='Validation Loss')
        plt.title('CNN-LSTM Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(cnn_lstm_history['train_acc'], label='Train Accuracy')
        plt.plot(cnn_lstm_history['val_acc'], label='Validation Accuracy')
        plt.title('CNN-LSTM Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('cnn_lstm_training_history.png')
        plt.close()
    
    # Evaluate CNN-LSTM model
    print("\n--- Evaluating CNN-LSTM Model ---")
    lstm_accuracy, lstm_precision, lstm_recall, lstm_f1, _ = evaluate_model(
        cnn_lstm_model, test_seq_loader, device, EMOTIONS
    )
    
    # Compare models
    print("\n--- Model Comparison ---")
    print(f"CNN Model - Accuracy: {cnn_accuracy:.4f}, F1-Score: {cnn_f1:.4f}")
    print(f"CNN-LSTM Model - Accuracy: {lstm_accuracy:.4f}, F1-Score: {lstm_f1:.4f}")
    
    # 4. Load YOLOv8 for face detection
    print("\n--- Loading YOLOv8 Face Detector ---")
    face_detector = load_face_detector()
    
    # 5. Run real-time emotion detection
    print("\n--- Running Real-time Emotion Detection ---")
    real_time_emotion_detection(face_detector, cnn_lstm_model, device)

if __name__ == "__main__":
    main()
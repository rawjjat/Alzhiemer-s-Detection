import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import (
    Input, Dense, Flatten, Reshape, LayerNormalization, MultiHeadAttention, 
    Conv2D, MaxPooling2D, Dropout, BatchNormalization
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from PIL import Image

# Define hyperparameters
image_size = 128
patch_size = 16
num_classes = 4  # Assuming 4 classes for dementia classification
hidden_dim = 64  # Reduced hidden dimensions for a lightweight model
num_layers = 2  # Reduced number of transformer layers
num_heads = 3  # Reduced number of attention heads

IMAGE_SIZE = (128, 128)
BATCH_SIZE = 32

# Data preprocessing
data_dir = 'C:/Users/rajat/OneDrive/Desktop/thesis/dataset/data_aug'
non_demented_dir = os.path.join(data_dir, 'non_demented')
mild_demented_dir = os.path.join(data_dir, 'mild_demented')
moderate_demented_dir = os.path.join(data_dir, 'moderate_demented')
very_mild_demented_dir = os.path.join(data_dir, 'very_mild_demented')

image_paths = []
labels = []

# Verify directory paths
for directory in [non_demented_dir, mild_demented_dir, moderate_demented_dir, very_mild_demented_dir]:
    if not os.path.exists(directory):
        print(f'Directory not found: {directory}')
    else:
        print(f'Directory exists: {directory} with {len(os.listdir(directory))} files')


for directory in [non_demented_dir, mild_demented_dir, moderate_demented_dir, very_mild_demented_dir]:
    for filename in os.listdir(directory)[:5000]:
        image_path = os.path.join(directory, filename)
        image_paths.append(image_path)
        if directory == non_demented_dir:
            labels.append(0)
        elif directory == mild_demented_dir:
            labels.append(1)
        elif directory == moderate_demented_dir:
            labels.append(2)
        elif directory == very_mild_demented_dir:
            labels.append(3)
            
import numpy as np
from PIL import Image

# Define a function to convert grayscale images to RGB
def convert_to_rgb(img):
    if img.ndim == 2:  # If image is grayscale
        img_rgb = np.stack((img,) * 3, axis=-1)  # Convert grayscale to RGB
    elif img.shape[-1] == 1:  # If image has a single channel
        img_rgb = np.concatenate([img] * 3, axis=-1)  # Add two more channels to make it RGB
    else:
        img_rgb = img  # Image is already in RGB format
    return img_rgb

X = []
for image_path in image_paths:
    try:
        img = Image.open(image_path)
        img = img.resize((128, 128))
        img = np.array(img)
        
        # Convert grayscale images to RGB
        img_rgb = convert_to_rgb(img)
        
        if img_rgb.shape == (128, 128, 3):
            X.append(img_rgb)
        else:
            print(f'Skipped image with unexpected shape: {img_rgb.shape}')
    except Exception as e:
        print(f'Error loading image {image_path}: {e}')


X = np.array(X)
y = np.array(labels)

# Debugging: Check shapes of X and y
print(f'Shape of X: {X.shape}')
print(f'Shape of y: {y.shape}')

encoder = OneHotEncoder()
y_encoded = encoder.fit_transform(y.reshape(-1, 1)).toarray()

# Debugging: Check shape of y_encoded
print(f'Shape of y_encoded: {y_encoded.shape}')

# Splitting the data
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, shuffle=True)

def build_lightweight_vit_with_cnn(image_size, patch_size, num_classes, hidden_dim, num_heads, num_layers):
    input_layer = Input(shape=(image_size, image_size, 3))  # RGB images

    # Recommended CNN Feature Extractor
    x = Conv2D(filters=32, kernel_size=2, padding='same', activation='relu')(input_layer)
    x = Conv2D(filters=32, kernel_size=2, padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
    x = Dropout(0.25)(x)
    
    x = Conv2D(filters=64, kernel_size=2, padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
    x = Dropout(0.25)(x)
    
    x = Conv2D(filters=128, kernel_size=2, padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
    x = Dropout(0.25)(x)
    
    # Flatten the CNN output
    cnn_features = Flatten()(x)

    # Adjust dimensions for the ViT model
    num_patches_side = image_size // patch_size
    num_patches_total = num_patches_side * num_patches_side
    cnn_features = Dense(num_patches_total * hidden_dim, activation='relu')(cnn_features)
    cnn_features = Reshape((num_patches_total, hidden_dim))(cnn_features)

    # Positional embeddings
    position_embeddings = tf.keras.layers.Embedding(input_dim=num_patches_total, output_dim=hidden_dim)(tf.range(num_patches_total))
    position_embeddings = tf.expand_dims(position_embeddings, axis=0)

    # Add positional embeddings
    patches = cnn_features + position_embeddings

    # Lightweight Transformer Encoder
    for _ in range(num_layers):
        # Layer normalization
        x1 = LayerNormalization(epsilon=1e-6)(patches)
        # Multi-head attention
        attention_output = MultiHeadAttention(num_heads=num_heads, key_dim=hidden_dim)(x1, x1)
        # Add & Norm
        x2 = x1 + attention_output
        # Layer normalization
        x3 = LayerNormalization(epsilon=1e-6)(x2)
        # Feedforward network
        ffn_output = Dense(hidden_dim, activation='relu')(x3)
        ffn_output = Dense(hidden_dim)(ffn_output)
        # Add & Norm
        patches = x3 + ffn_output

    # Classification head
    x = Flatten()(patches)
    output = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=input_layer, outputs=output)
    return model

# Build and compile the model
vit_model = build_lightweight_vit_with_cnn(image_size, patch_size, num_classes, hidden_dim, num_heads, num_layers)
vit_model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
vit_model.summary()

# Callbacks for saving the best model and early stopping
checkpoint = ModelCheckpoint('best_vit_model.h5', monitor='val_loss', save_best_only=True, mode='min')
early_stopping = EarlyStopping(monitor='val_loss', patience=10, mode='min')

# Training the model
history = vit_model.fit(
    X_train, y_train,
    epochs=15,
    validation_split=0.2,
    
    callbacks=[checkpoint, early_stopping],
    batch_size=BATCH_SIZE
)

# Load the best model
vit_model.load_weights('best_vit_model.h5')

# Evaluate the model on validation data
test_loss, test_accuracy = vit_model.evaluate(X_test, y_test)
print(f'Test loss: {test_loss}')
print(f'Test accuracy: {test_accuracy}')

# Generate a classification report
y_pred = vit_model.predict(X_test)
y_pred_classes = y_pred.argmax(axis=-1)
y_true = y_test.argmax(axis=-1)

print(classification_report(y_true, y_pred_classes, target_names=['non_demented', 'mild_demented', 'moderate_demented', 'very_mild_demented']))

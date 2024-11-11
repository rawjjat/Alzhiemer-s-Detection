import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# Define data generators for training and validation sets
IMAGE_SIZE = (224, 224)  # Original size for MobileNetV2
LARGER_IMAGE_SIZE = (256, 256)  # Larger size for input images
BATCH_SIZE = 32
NUM_CLASSES = 4
base_path = "C:/Users/rajat/OneDrive/Desktop/thesis/dataset/augmented"

train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2
)

train_generator = train_datagen.flow_from_directory(
    base_path,
    target_size=LARGER_IMAGE_SIZE,  # Use larger image size
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

val_generator = train_datagen.flow_from_directory(
    base_path,
    target_size=LARGER_IMAGE_SIZE,  # Use larger image size
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

# Define MobileNetV2 architecture
def mobilenetv2(input_shape):
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(32, (3, 3), strides=(2, 2), padding='same', activation='relu')(inputs)
    x = layers.DepthwiseConv2D((3, 3), padding='same', activation='relu')(x)
    x = layers.Conv2D(16, (1, 1), padding='same', activation='relu')(x)
    x = layers.DepthwiseConv2D((3, 3), strides=(2, 2), padding='same', activation='relu')(x)
    x = layers.Conv2D(24, (1, 1), padding='same', activation='relu')(x)
    x = layers.DepthwiseConv2D((3, 3), padding='same', activation='relu')(x)
    x = layers.Conv2D(24, (1, 1), padding='same', activation='relu')(x)
    x = layers.DepthwiseConv2D((3, 3), strides=(2, 2), padding='same', activation='relu')(x)
    x = layers.Conv2D(32, (1, 1), padding='same', activation='relu')(x)
    x = layers.DepthwiseConv2D((3, 3), padding='same', activation='relu')(x)
    x = layers.Conv2D(32, (1, 1), padding='same', activation='relu')(x)
    x = layers.DepthwiseConv2D((3, 3), strides=(2, 2), padding='same', activation='relu')(x)
    x = layers.Conv2D(64, (1, 1), padding='same', activation='relu')(x)
    x = layers.DepthwiseConv2D((3, 3), padding='same', activation='relu')(x)
    x = layers.Conv2D(64, (1, 1), padding='same', activation='relu')(x)
    x = layers.DepthwiseConv2D((3, 3), strides=(2, 2), padding='same', activation='relu')(x)
    x = layers.Conv2D(96, (1, 1), padding='same', activation='relu')(x)
    x = layers.DepthwiseConv2D((3, 3), padding='same', activation='relu')(x)
    x = layers.Conv2D(96, (1, 1), padding='same', activation='relu')(x)
    x = layers.DepthwiseConv2D((3, 3), strides=(2, 2), padding='same', activation='relu')(x)
    x = layers.Conv2D(160, (1, 1), padding='same', activation='relu')(x)
    x = layers.DepthwiseConv2D((3, 3), padding='same', activation='relu')(x)
    x = layers.Conv2D(160, (1, 1), padding='same', activation='relu')(x)
    x = layers.DepthwiseConv2D((3, 3), strides=(2, 2), padding='same', activation='relu')(x)
    x = layers.Conv2D(320, (1, 1), padding='same', activation='relu')(x)
    x = layers.GlobalAveragePooling2D()(x)
    return Model(inputs=inputs, outputs=x)

# Define the Vision Transformer model with MobileNetV2 feature extractor
def VisionTransformer(image_size, patch_size, num_classes, d_model, num_heads, mlp_dim, num_layers):
    inputs = layers.Input(shape=image_size + (3,))

    # Use MobileNetV2 model for feature extraction
    mobilenet_model = mobilenetv2(IMAGE_SIZE)
    features = mobilenet_model(inputs)

    # Calculate the number of patches
    num_patches = (features.shape[1] // patch_size) * (features.shape[2] // patch_size)

    # Patching
    patches = layers.Reshape((num_patches, d_model))(features)
    
    # Positional embeddings
    positions = tf.range(start=0, limit=num_patches, delta=1)
    positions = layers.Embedding(input_dim=num_patches, output_dim=d_model)(positions)
    
    # Add positional embeddings to patches
    embeddings = layers.Add()([positions, patches])
    patches = layers.Reshape((num_patches, d_model))(patches)
    
    # Positional embeddings
    positions = tf.range(start=0, limit=num_patches, delta=1)
    positions = layers.Embedding(input_dim=num_patches, output_dim=d_model)(positions)
    
    # Add positional embeddings to patches
    embeddings = layers.Add()([positions, patches])
    
    # Transformer encoder
    for _ in range(num_layers):
        # Multi-head self-attention
        attention_output = layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model // num_heads)(embeddings, embeddings)
        attention_output = layers.LayerNormalization(epsilon=1e-6)(attention_output + embeddings)
        
        # MLP
        mlp_output = layers.Dense(units=mlp_dim, activation="relu")(attention_output)
        mlp_output = layers.Dense(units=d_model)(mlp_output)
        embeddings = layers.LayerNormalization(epsilon=1e-6)(mlp_output + attention_output)
    
    # Class token
    class_token = layers.Embedding(input_dim=1, output_dim=d_model)(tf.constant([0]))
    class_token = tf.tile(class_token, [tf.shape(inputs)[0], 1])  # Tile class token to match the shape of embeddings
    class_token = tf.expand_dims(class_token, axis=1)  # Expand dims to match the shape of embeddings along the concatenation axis
    embeddings = layers.Concatenate(axis=1)([class_token, embeddings])
    
    # Classifier head
    outputs = layers.Dense(units=num_classes, activation="softmax")(embeddings[:, 0, :])
    
    return Model(inputs=inputs, outputs=outputs)

# Parameters
PATCH_SIZE = 16
D_MODEL = 512
NUM_HEADS = 8
MLP_DIM = 1024
NUM_LAYERS = 6

# Create ViT model
vit_model = VisionTransformer(
    image_size=LARGER_IMAGE_SIZE,
    patch_size=PATCH_SIZE,
    num_classes=NUM_CLASSES,
    d_model=D_MODEL,
    num_heads=NUM_HEADS,
    mlp_dim=MLP_DIM,
    num_layers=NUM_LAYERS
)

# Compile the model
vit_model.compile(optimizer=Adam(learning_rate=1e-4),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

# Train the ViT model using the extracted features
vit_history = vit_model.fit(
    train_generator, 
    epochs=20, 
    validation_data=val_generator
)
# Plot training history
plt.plot(vit_history.history['accuracy'], label='accuracy')
plt.plot(vit_history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Evaluate the model on validation data
val_loss, val_accuracy = vit_model.evaluate(val_generator)
print(f'Validation loss: {val_loss}')
print(f'Validation accuracy: {val_accuracy}')

# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 23:59:16 2024

@author: rajat
"""

import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.optimizers import Adam

# Define ViT model
def VisionTransformer(image_size, patch_size, num_classes, d_model, num_heads, mlp_dim, num_layers):
    inputs = layers.Input(shape=image_size + (3,))
    
    # Patching
    patches = layers.Conv2D(filters=d_model, kernel_size=patch_size, strides=patch_size, padding="valid")(inputs)
    num_patches = (image_size[0] // patch_size) * (image_size[1] // patch_size)
    patch_shape = patches.shape[1:3]
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
    class_token = tf.tile(class_token, [num_patches + 1, 1])  # Tile class token to match the shape of embeddings
    embeddings = layers.Concatenate(axis=1)([class_token, embeddings])
    
    # Classifier head
    outputs = layers.Dense(units=num_classes, activation="softmax")(embeddings[:, 0, :])
    
    return Model(inputs=inputs, outputs=outputs)

# Parameters
IMAGE_SIZE = (224, 224)
PATCH_SIZE = 16
NUM_CLASSES = 4
D_MODEL = 256
NUM_HEADS = 8
MLP_DIM = 512
NUM_LAYERS = 6

# Create ViT model
vit_model = VisionTransformer(
    image_size=IMAGE_SIZE,
    patch_size=PATCH_SIZE,
    num_classes=NUM_CLASSES,
    d_model=D_MODEL,
    num_heads=NUM_HEADS,
    mlp_dim=MLP_DIM,
    num_layers=NUM_LAYERS
)

# Compile the model
vit_model.compile(optimizer=Adam(learning_rate=1e-3),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

vit_model.summary()
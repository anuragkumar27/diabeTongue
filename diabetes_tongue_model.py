import tensorflow as tf
from tensorflow.keras import layers, models, applications
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc
import seaborn as sns

# Data loading and preprocessing
def create_data_generators(batch_size=32):
    train_datagen = ImageDataGenerator(
        rotation_range=20,
        zoom_range=0.2,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        vertical_flip=True,
        preprocessing_function=tf.keras.applications.efficientnet.preprocess_input
    )

    test_datagen = ImageDataGenerator(preprocessing_function=tf.keras.applications.efficientnet.preprocess_input)

    train_generator = train_datagen.flow_from_directory(
        'data/preprocessedcropped/train',
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='binary'
    )

    valid_generator = test_datagen.flow_from_directory(
        'data/preprocessedcropped/valid',
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='binary'
    )

    test_generator = test_datagen.flow_from_directory(
        'data/preprocessedcropped/test',
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='binary',
        shuffle=False
    )

    return train_generator, valid_generator, test_generator

# Model architecture
def build_hybrid_model(input_shape=(224, 224, 3)):
    # CNN Backbone
    base_model = applications.EfficientNetB0(
        include_top=False,
        weights='imagenet',
        input_shape=input_shape
    )
    base_model.trainable = False  # Freeze initially

    inputs = layers.Input(shape=input_shape)
    x = base_model(inputs, training=False)
    x = layers.SpatialDropout2D(0.3)(x)
    
    # Attention mechanism
    attn = layers.Conv2D(1280, 1, activation='relu')(x)
    attn = layers.Softmax(axis=-1)(attn)
    x = layers.Multiply()([x, attn])
    
    # Transformer component
    x = layers.Reshape((-1, 1280))(x)
    x = layers.MultiHeadAttention(num_heads=8, key_dim=64)(x, x)
    x = layers.LayerNormalization()(x)
    
    # Expand dimensions to make it 4D for GlobalAveragePooling2D
    x = layers.Reshape((7, 7, 1280))(x)
    
    x = layers.GlobalAveragePooling2D()(x)
    
    # Classification head
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)

    model = models.Model(inputs, outputs)
    
    return model

# Training function
def train_model():
    # Create data generators
    train_gen, valid_gen, test_gen = create_data_generators(batch_size=32)
    
    # Build model
    model = build_hybrid_model()
    
    # Initial learning rate configuration
    initial_learning_rate = 1e-3
    optimizer = tf.keras.optimizers.Adam(learning_rate=initial_learning_rate)
    
    # Compile model
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )

    # Phase 1 callbacks (initial training)
    phase1_callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint('best_model.h5', save_best_only=True),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=2,
            min_lr=1e-6
        )
    ]

    # Initial training with frozen base
    history = model.fit(
        train_gen,
        validation_data=valid_gen,
        epochs=30,
        callbacks=phase1_callbacks
    )

    # Phase 2 (fine-tuning)
    model.layers[1].trainable = True
    
    # New optimizer for fine-tuning
    fine_tune_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5)
    
    # Phase 2 callbacks (without learning rate schedulers)
    phase2_callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=3),
        tf.keras.callbacks.ModelCheckpoint('fine_tuned_model.h5', save_best_only=True)
    ]

    model.compile(
        optimizer=fine_tune_optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy', 'auc']
    )

    history_fine = model.fit(
        train_gen,
        validation_data=valid_gen,
        epochs=15,
        callbacks=phase2_callbacks  # Note: Different callbacks!
    )
    
    # Evaluate on test set
    test_loss, test_acc, test_auc = model.evaluate(test_gen)
    print(f'\nTest accuracy: {test_acc:.4f}')
    print(f'Test AUC: {test_auc:.4f}')
    
    # Generate predictions for evaluation
    y_pred = model.predict(test_gen)
    y_true = test_gen.classes
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'] + history_fine.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'] + history_fine.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'] + history_fine.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'] + history_fine.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.show()
    
    # Confusion matrix and ROC curve
    cm = confusion_matrix(y_true, (y_pred > 0.5).astype(int))
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    
    plt.subplot(1, 2, 2)
    plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.show()
    
    return model

# Main execution
if __name__ == '__main__':
    trained_model = train_model()
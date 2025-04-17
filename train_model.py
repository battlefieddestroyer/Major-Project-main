import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.efficientnet import EfficientNetB0, preprocess_input as efficientnet_preprocess_input
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input as vgg16_preprocess_input
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler, TensorBoard, ReduceLROnPlateau
import datetime

# Enable mixed precision training
tf.keras.mixed_precision.set_global_policy('mixed_float16')

# Data augmentation
train_datagen = ImageDataGenerator(
    preprocessing_function=efficientnet_preprocess_input,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(
    preprocessing_function=efficientnet_preprocess_input,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

train_generator = train_datagen.flow_from_directory(
    'train',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

val_generator = val_datagen.flow_from_directory(
    'valid',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

# Load EfficientNet model
base_model_efficientnet = EfficientNetB0(weights='imagenet', include_top=False)
x = base_model_efficientnet.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(train_generator.num_classes, activation='softmax', dtype='float32')(x)

model_efficientnet = Model(inputs=base_model_efficientnet.input, outputs=predictions)

# Print model summary
model_efficientnet.summary()

# Freeze the base model layers
for layer in base_model_efficientnet.layers:
    layer.trainable = False

# Compile the model
model_efficientnet.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
model_checkpoint = ModelCheckpoint('best_model_efficientnet.h5', save_best_only=True, monitor='val_loss')
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.00001)

def scheduler(epoch, lr):
    if epoch < 10:
        return lr
    else:
        return lr * tf.math.exp(-0.1)

lr_scheduler = LearningRateScheduler(scheduler)

# Train the model
model_efficientnet.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    validation_data=val_generator,
    validation_steps=val_generator.samples // val_generator.batch_size,
    epochs=20,
    callbacks=[early_stopping, model_checkpoint, tensorboard_callback, lr_scheduler, reduce_lr]
)

# Unfreeze some layers of the base model for fine-tuning
for layer in base_model_efficientnet.layers[-20:]:
    layer.trainable = True

# Recompile the model with a lower learning rate
model_efficientnet.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# Continue training the model
model_efficientnet.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    validation_data=val_generator,
    validation_steps=val_generator.samples // val_generator.batch_size,
    epochs=10,
    callbacks=[early_stopping, model_checkpoint, tensorboard_callback, lr_scheduler, reduce_lr]
)

# Save the trained model
model_efficientnet.save('plant_disease_efficientnet.keras')

# VGG16 Model
# Data augmentation
train_datagen_vgg16 = ImageDataGenerator(
    preprocessing_function=vgg16_preprocess_input,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_datagen_vgg16 = ImageDataGenerator(
    preprocessing_function=vgg16_preprocess_input,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

train_generator_vgg16 = train_datagen_vgg16.flow_from_directory(
    'train',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

val_generator_vgg16 = val_datagen_vgg16.flow_from_directory(
    'valid',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

# Load VGG16 model
base_model_vgg16 = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
x = base_model_vgg16.output
x = GlobalAveragePooling2D()(x)
x = Dense(1500, activation='relu')(x)
x = Dropout(0.4)(x)
predictions = Dense(train_generator_vgg16.num_classes, activation='softmax')(x)

model_vgg16 = Model(inputs=base_model_vgg16.input, outputs=predictions)

# Print model summary
model_vgg16.summary()

# Freeze the base model layers
for layer in base_model_vgg16.layers:
    layer.trainable = False

# Compile the model
model_vgg16.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model_vgg16.fit(
    train_generator_vgg16,
    steps_per_epoch=train_generator_vgg16.samples // train_generator_vgg16.batch_size,
    validation_data=val_generator_vgg16,
    validation_steps=val_generator_vgg16.samples // val_generator_vgg16.batch_size,
    epochs=10,
    callbacks=[early_stopping, model_checkpoint, tensorboard_callback, lr_scheduler, reduce_lr]
)

# Save the trained model
model_vgg16.save('plant_disease_vgg16.keras')

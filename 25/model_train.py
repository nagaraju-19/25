import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_data = train.flow_from_directory(
    "dataset",
    target_size=(224,224),
    batch_size=32,
    class_mode="categorical",
    subset="training"
)

val_data = train.flow_from_directory(
    "dataset",
    target_size=(224,224),
    batch_size=32,
    class_mode="categorical",
    subset="validation"
)

model = tf.keras.models.Sequential([

tf.keras.layers.Input(shape=(224,224,3)),

tf.keras.layers.Conv2D(32,(3,3),activation="relu"),
tf.keras.layers.MaxPooling2D(2,2),

tf.keras.layers.Conv2D(64,(3,3),activation="relu"),
tf.keras.layers.MaxPooling2D(2,2),

tf.keras.layers.Conv2D(128,(3,3),activation="relu"),
tf.keras.layers.MaxPooling2D(2,2),

tf.keras.layers.Flatten(),

tf.keras.layers.Dense(256,activation="relu"),
tf.keras.layers.Dropout(0.5),

tf.keras.layers.Dense(train_data.num_classes,activation="softmax")

])

model.compile(
optimizer="adam",
loss="categorical_crossentropy",
metrics=["accuracy"]
)

model.fit(
train_data,
validation_data=val_data,
epochs=10
)

model.save("model/plant_model.h5")
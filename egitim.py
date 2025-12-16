import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib
matplotlib.use("TkAgg")  # veya "Agg" sadece kaydetmek için
import matplotlib.pyplot as plt


# =========================
# AYARLAR
# =========================
DATASET_DIR = r"C:\Users\Eren\Desktop\fashion-dataset\dataset2"
IMG_SIZE = (128, 128)
BATCH_SIZE = 8
EPOCHS = 15

# =========================
# DATASET
# =========================
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    DATASET_DIR + "/train",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical"
)

test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    DATASET_DIR + "/test",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical",
    shuffle=False
)

class_names = train_ds.class_names
NUM_CLASSES = len(class_names)
print("Sınıf sayısı:", NUM_CLASSES)

# =========================
# NORMALIZATION
# =========================
normalization = layers.Rescaling(1./255)
train_ds = train_ds.map(lambda x, y: (normalization(x), y))
test_ds = test_ds.map(lambda x, y: (normalization(x), y))

# =========================
# CNN MODEL
# =========================
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation="relu", input_shape=(128,128,3)),
    layers.MaxPooling2D(2,2),

    layers.Conv2D(64, (3,3), activation="relu"),
    layers.MaxPooling2D(2,2),

    layers.Conv2D(128, (3,3), activation="relu"),
    layers.MaxPooling2D(2,2),

    layers.Flatten(),
    layers.Dense(256, activation="relu"),
    layers.Dropout(0.5),
    layers.Dense(NUM_CLASSES, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# =========================
# EĞİTİM
# =========================
history = model.fit(
    train_ds,
    validation_data=test_ds,
    epochs=EPOCHS
)

# =========================
# GRAFİKLER
# =========================
plt.plot(history.history["accuracy"], label="Train Acc")
plt.plot(history.history["val_accuracy"], label="Val Acc")
plt.legend()
plt.title("Accuracy")
plt.show()

plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Val Loss")
plt.legend()
plt.title("Loss")
plt.show()

# =========================
# MODEL KAYDET
# =========================
model.save("fashion_cnn_30class.h5")
print("✅ Model kaydedildi: fashion_cnn_30class.h5")

import tensorflow as tf
import os
x=os.walk(r"./data")
x=list(x)
entries={}
it = 0
for i in range(1,len(x)) :

    entries[i]=x[i][0].split("\\")[-1]
    
# print(entries)
def makeDefaultHiddenLayers(inputs):
        cnn = tf.keras.layers.Conv2D(16, (3, 3), padding="same",activation="relu")(inputs)
        # cnn = tf.keras.layers.core.Activation("relu")(cnn)
        cnn = tf.keras.layers.BatchNormalization(axis=-1)(cnn)
        cnn = tf.keras.layers.MaxPooling2D(pool_size=(3, 3))(cnn)
        cnn = tf.keras.layers.Dropout(0.25)(cnn)
        cnn = tf.keras.layers.Conv2D(32, (3, 3), padding="same")(cnn)
        cnn = tf.keras.layers.Activation("relu")(cnn)
        cnn = tf.keras.layers.BatchNormalization(axis=-1)(cnn)
        cnn = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(cnn)
        cnn = tf.keras.layers.Dropout(0.25)(cnn)
        cnn = tf.keras.layers.Conv2D(32, (3, 3), padding="same")(cnn)
        cnn = tf.keras.layers.Activation("relu")(cnn)
        cnn = tf.keras.layers.BatchNormalization(axis=-1)(cnn)
        cnn = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(cnn)
        cnn = tf.keras.layers.Dropout(0.25)(cnn)
        return cnn
def makeModal(inputs,class_dict):
    cnn=makeDefaultHiddenLayers(inputs)
    cnn = tf.keras.layers.Flatten()(cnn)
    cnn = tf.keras.layers.Dense(128)(cnn)
    cnn = tf.keras.layers.Activation("relu")(cnn)
    cnn = tf.keras.layers.BatchNormalization()(cnn)
    cnn = tf.keras.layers.Dropout(0.5)(cnn)
    cnn = tf.keras.layers.Dense(class_dict)(cnn)
    cnn = tf.keras.layers.Activation("softmax", name="Bark_Output")(cnn)

    return cnn
def assemble(width,height,class_dict):
    inputs = tf.keras.layers.Input(shape=(height, width, 3))
    cnnOutputLayers=makeModal(inputs,class_dict)
    model=tf.keras.models.Model(inputs=inputs,outputs = [cnnOutputLayers],name="face_net")
    return model

X=assemble(303, 404,50)
# print(X.output_shape)
dataset = tf.keras.utils.image_dataset_from_directory(
    directory='data/',
    labels='inferred',
    label_mode='categorical',
    batch_size=32,
    image_size=(404, 303),
    shuffle=True,
    seed=123,
    validation_split=0.2,
    subset="both")

X.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
X.fit(x = dataset[0], validation_data = dataset[1], epochs = 25)
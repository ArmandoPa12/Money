#import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt

project_dir = os.path.dirname(os.path.abspath(__file__))

# Directorio de guardado de los archivos .npz
save_dir = os.path.join(project_dir, 'saved_data')

# Cargar los datos guardados
#train_path = os.path.join(save_dir, 'train_data.npz')
val_path = os.path.join(save_dir, 'val_data.npz')

val_data = np.load(val_path)
images = val_data['X_val']
label = val_data['y_val']

print(val_data.files)

num_images = 100
plt.figure(figsize=(10,5))



for i in range(num_images):
    plt.subplot(1,num_images,i+1)
    plt.imshow(images[i], cmap='gray')
    plt.title(f"Label: {label[i]}")
    plt.axis('off')

plt.show()

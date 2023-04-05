# Computer Vision  Задание 2
На основе ноутбука из лекции, настроить и обучить полносвязную прямую нейронную сеть. Классифицировать объекты из датасета cifar10 (https://keras.io/api/datasets/cifar10/). Желательно добиться как можно большего значения accuracy. Можно менять функции активации, функцию ошибки (loss), типы слоев, количество нейронов в слоях, количество слоев, оптимизаторы, количество эпох и т.д. Нельзя менять метрику и тип нейронной сети. 

Для начала мы загрузили датасет и модель в ноутук, предобработали данные и посмотрели примеры.

**Пример изображения в обучающем датасете**
![image info](https://github.com/misshp11/CV2/blob/main/img/изображение_2023-04-05_005556395.png) 
![image info](https://github.com/misshp11/CV2/blob/main/img/изображение_2023-04-05_005945826.png) 

Затем мы определили нашу модель и прописали какие слои будут в нашей модели.
```python
model = keras.Sequential([
                          keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
                          keras.layers.MaxPooling2D(pool_size=(2, 2)),
                          keras.layers.Conv2D(64, 3, activation="relu"),
                          keras.layers.MaxPooling2D(pool_size=(2, 2)),
                          keras.layers.Conv2D(128, 3, activation="relu"),
                          keras.layers.Flatten(),
                          keras.layers.Dense(256, activation='relu'),
                          keras.layers.Dense(10, activation="softmax")
])
```

Скомпилировали нашу модель, прописали какой используем оптимизатор и функцию ошибки, метрика у нас исталась неизменной.
```python
model.compile(optimizer=tf.keras.optimizers.Adamax(learning_rate=0.002), loss='SparseCategoricalCrossentropy', metrics=['accuracy'])
```

Ниже представлено сводное представление нашей модели.
![image info](https://github.com/misshp11/CV2/blob/main/img/изображение_2023-04-05_011104903.png)                                                   

Дальше у нас пошло обучение модели.
![image info](https://github.com/misshp11/CV2/blob/main/img/изображение_2023-04-05_011342097.png) 

Проверим на тестовой выборке насколько хорошо обучилась наша модель.
```python
test_loss, test_acc = model.evaluate(x_test, y_test)
print('Test loss:', test_loss)
print('Test accuracy:', test_acc)
```
<code>313/313 [==============================] - 6s 19ms/step - loss: 1.3662 - accuracy: 0.7344
Test loss: 1.3662052154541016
Test accuracy: 0.7343999743461609
</code> 
Точность составляет 0.7343999743461609 или же ~73%

Распознаем изображения для тестирования и просматриваем.
![image info](https://github.com/misshp11/CV2/blob/main/img/изображение_2023-04-05_012421835.png)  

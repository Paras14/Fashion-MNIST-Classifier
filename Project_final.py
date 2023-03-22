import sys, os, numpy as np
import imageio.v3 as imageio
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
import os.path as path
from numpy import unravel_index
# from sklearn.metrics import confusion_matrix

if __name__ == "__main__":

    if (path.exists("X_samples.npy") == False):
        X_List = []
        T_List = []

        for f in os.listdir(sys.argv[1]):
            # print ("File ", f ," class is ", int(f[0]))
            img = imageio.imread(str(sys.argv[1])+"/"+f)
            X_List.append(img)

            label = int(f[0])

            label_hot_encode = np.zeros(10)
            label_hot_encode[label] = 1 
            T_List.append(label_hot_encode)


        X = np.array(X_List)
        T = np.array(T_List)
        np.save('X_samples',X)
        np.save('T_labels',T)
        print('Data load with file names X_samples.npy and T_labels.npy')

    else:
        X = np.load('X_samples.npy')
        T = np.load('T_labels.npy')


    print("X.shape: ", X.shape)
    print("T.shape: ",T.shape)
    X = X.reshape(-1, 28, 28, 1)

    num_examples_per_class = np.sum(T, axis=0)

    print("Number of examples per classes before oversampling class 1", num_examples_per_class)
    labels = np.argmax(T, axis=1)
    labels = labels + 1

    # histogram of the labels
    plt.hist(labels, 10)
    plt.xlabel('Class')
    plt.ylabel('Number of Examples')
    plt.title('Distribution of Examples across Classes before oversampling')
    plt.show()

    # Number of examples in class 1 or index 0
    num_class_0_examples = num_examples_per_class[0]

    # Number of examples in majority class
    num_majority_class_examples = np.max(num_examples_per_class)

    # Index positions of class 1 examples .... array exists on Tuple's first index 0 thats why [0]
    class_0_indices = np.where(np.argmax(T, axis=1) == 0)[0]

    # Number of duplicate examples needed
    num_duplicates_needed = num_majority_class_examples - num_class_0_examples
    num_duplicates_needed = int(num_duplicates_needed)

    # Empty array to store the duplicate examples
    duplicate_X = np.empty((num_duplicates_needed, 28, 28, 1))
    duplicate_T = np.empty((num_duplicates_needed, 10))

    # Index position for next duplicate
    next_index = 0

    # Fill the duplicate arrays with duplicate examples
    for i in range(num_duplicates_needed):
        duplicate_X[next_index] = X[int(class_0_indices[int(i % num_class_0_examples)])]
        duplicate_T[next_index] = T[int(class_0_indices[int(i % num_class_0_examples)])]
        next_index += 1

    # Concatenate the original and duplicate arrays
    X_oversampled = np.concatenate((X, duplicate_X))
    T_oversampled = np.concatenate((T, duplicate_T))

    num_examples_per_class = np.sum(T_oversampled, axis=0)
    print("Number of examples per classes after oversampling class 1", num_examples_per_class)
    labels = np.argmax(T_oversampled, axis=1)
    labels = labels + 1
    # another Histogram with balanced classes
    plt.hist(labels, 10)
    plt.xlabel('Class')
    plt.ylabel('Number of Examples')
    plt.title('Distribution of Examples across Classes after oversampling')
    plt.show()

    # Noise Removal using Median Filter
    clipped_array = np.clip(X_oversampled, 0, 255)
    X_oversampled = clipped_array.astype(np.uint8)

    #Displaying before medianBlur
    fig, axs = plt.subplots(1, 3, figsize=(10,10))
    axs[0].imshow(X_oversampled[24], cmap='gray')
    axs[1].imshow(X_oversampled[54211], cmap='gray')
    axs[2].imshow(X_oversampled[24211], cmap='gray')
    plt.show()

    for i in range(0, X_oversampled.shape[0]):
        img_tmp = X_oversampled[i]
        highest_index = unravel_index(img_tmp.argmax(), img_tmp.shape)
        window = img_tmp[(highest_index[0]-2):(highest_index[0]+3),(highest_index[1]-2):(highest_index[1]+3)]
        avg = (np.sum(window) - img_tmp[highest_index]) / 24
        avg = avg.astype(np.uint8)

        if(i==24211):
            # print(img_tmp[12, 13])
            # print(img_tmp.shape)
            fig, axs = plt.subplots(1, 1, figsize=(10,10))
            axs.imshow(img_tmp, cmap='gray')
            plt.show()
            
        img_tmp[highest_index[0], highest_index[1]] = avg
        # img_tmp[14,14] = np.mean(img_tmp[(highest_index[0]-2):(highest_index[0]+3),(highest_index[1]-2):(highest_index[1]+3)]).astype(np.uint8)
        # img_tmp[14,14] = 21
            
        if(i==24211):
            # print(img_tmp[12, 13])
            # print(img_tmp.shape)
            fig, axs = plt.subplots(1, 1, figsize=(10,10))
            axs.imshow(img_tmp, cmap='gray')
            plt.show()

        X_oversampled[i] = np.reshape(img_tmp, (28, 28, 1))


    #Displaying after medianBlur
    fig, axs = plt.subplots(1, 3, figsize=(10,10))
    axs[0].imshow(X_oversampled[24], cmap='gray')
    axs[1].imshow(X_oversampled[54211], cmap='gray')
    axs[2].imshow(X_oversampled[24211], cmap='gray')
    plt.show()

    # Shuffling Data
    rdm = np.arange(X_oversampled.shape[0])
    np.random.shuffle(rdm)
    X_oversampled_shuffled = X_oversampled[rdm]
    T_oversampled_shuffled = T_oversampled[rdm]

    # Splitting Training and Testing Cases

    #Taking 20% of samples for testCases
    N = int(X_oversampled_shuffled.shape[0])

    start_for_testCases = N - int(0.2 * N)
    end_for_testCases = N

    # Start and end index for train cases
    start_for_trainCases = 0
    end_for_trainCases = start_for_testCases

    testCases_X = X_oversampled_shuffled[start_for_testCases:end_for_testCases]
    testCases_T = T_oversampled_shuffled[start_for_testCases:end_for_testCases]

    X_train = X_oversampled_shuffled[start_for_trainCases:end_for_trainCases]
    print("X_train.shape : ",X_train.shape)
    T_train = T_oversampled_shuffled[start_for_trainCases:end_for_trainCases]
    print("T_train.shape : ",T_train.shape)


    # Training Code

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv2D(filters=16, kernel_size=2, padding="same", activation="relu", input_shape=(28, 28, 1))) #(28,28,16)
    model.add(tf.keras.layers.MaxPooling2D(pool_size=2))#(14,14,16)
    model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=2, padding="same", activation="relu"))#(14,14,32)
    model.add(tf.keras.layers.MaxPooling2D(pool_size=2))#(7,7,32)
    model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=2, padding="same", activation="relu"))#(7,7,64)
    model.add(tf.keras.layers.MaxPooling2D(pool_size=2))#(4,4,64)
    model.add(tf.keras.layers.Flatten())#4*4*64 = (1024,)
    model.add(tf.keras.layers.Dense(10))#(10,)
    model.add(tf.keras.layers.Softmax())#(10,)

    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss=tf.keras.losses.CategoricalCrossentropy(),
                  metrics=[tf.keras.metrics.CategoricalAccuracy()])

    model.fit(x=X_train, y=T_train, epochs=9, shuffle=True)

    class_error = model.evaluate(x=testCases_X.reshape(-1, 28, 28), y=testCases_T);  # compute f(X)
    print("classification error on testCases_X is ", class_error)
    
    #Confusion Matrix
    prediction = model.predict(testCases_X)
    print('testCases_X: ', testCases_X)
    print('prediction: ',prediction.shape)
    normalized = np.arange(0, prediction.shape[0])
    print('normalized: ', normalized)
    target = np.arange(0, testCases_T.shape[0])
    print('target: ',target)
    confusion_matrix = np.zeros((10, 10), dtype=int)

    # diag_line = np.dot(testCases_X, prediction)

    normalized = np.argmax(prediction, axis=1)
    target = np.argmax(testCases_T, axis=1)

    np.add.at(confusion_matrix, (target, normalized), 1)



    # for i in range(0, prediction.shape[0]):
    #     normalized[i] = np.argmax(prediction[i])
    #     target[i] = np.argmax(testCases_T[i])
    #     confusion_matrix[target[i]][normalized[i]] += 1

    print("Confusion Matrix")
    print(confusion_matrix)

    # labels = np.unique(testCases_T)
    # no_of_classes = len(labels)

    # val = np.bincount(testCases_T * no_of_classes + prediction)
    # conf_mat = val.reshape(no_of_classes, no_of_classes)
    # print(conf_mat)



#!/usr/bin/env python
# coding: utf-8

# In[ ]:


data = np.load('data.npy')
data = data.item()

number_of_samples = 5
Xtrain = data.get('Xtrain')
Xtest = data.get('Xtest')
Ytrain = data.get('Ytrain')
Ytest = data.get('Ytest')
Sample_selection = np.random.choice(3000, number_of_samples)
images = []
for i in range(number_of_samples):
    images.append(Xtrain[Sample_selection[i]])
j = 0
for image in images:
    i = 28
    new_image = image[0:28]
    while(i<783):
        new_image = np.vstack([new_image, image[i:(i+28)]])
        i += 28
    images[j] = new_image
    j += 1
    
w=10
h=10
fig=plt.figure(figsize=(8, 8))
columns = 1
rows = 5
ax = []
for i in range(1, columns*rows +1):
    ax.append(fig.add_subplot(rows, columns, i))
    ax[-1].set_title('{:01d}'.format(Ytrain[Sample_selection[i-1]]))  # set title
    plt.imshow(images[i-1])
plt.show()

from sklearn.neighbors import KNeighborsClassifier


number_of_samples = 1000
Sample_selection = np.random.choice(3000, number_of_samples)
images = []
for i in range(number_of_samples):
    images.append(Xtrain[Sample_selection[i]])
    
neigh = KNeighborsClassifier(n_neighbors=3)
Y = []
for i in range(number_of_samples):
    Y.append(Ytrain[Sample_selection[i]])

neigh.fit(images, Y) 
Y_test_pred = neigh.predict(Xtest)
error = np.linalg.norm(Y_test_pred - Ytest)

error_array = []
n_neighbors_array = [1,3,5,7,9]
for n__neighbors in n_neighbors_array:
    neigh = KNeighborsClassifier(n_neighbors=n__neighbors)
    neigh.fit(images, Y) 
    Y_test_pred = neigh.predict(Xtest)
    error = np.linalg.norm(Y_test_pred - Ytest)
    error_array.append(error/number_of_samples)
  
plt.plot(n_neighbors_array, error_array)

n_neighbors_array = [1,3,5,7,9]
n_samples_array = [1000,1400,1800,2200,2600,3000]
n_neighbors_error_array = []
for n__neighbors in n_neighbors_array:
    n_samples_error_array = []
    for n in n_samples_array:
        images = []
        Y = []
        for i in range(n):
            images.append(Xtrain[i])
            Y.append(Ytrain[i])
        neigh = KNeighborsClassifier(n_neighbors=n__neighbors)
        neigh.fit(images, Y) 
        Y_test_pred = neigh.predict(Xtest)
        error = np.linalg.norm(Y_test_pred - Ytest)
        n_samples_error_array.append(error/n)
    n_neighbors_error_array.append(n_samples_error_array)

plt.figure(0, figsize=(12, 8))
colors = ['green', 'purple', 'yellow', 'brown', 'black']
labels = ['1', '3', '5', '7', '9']
indices = [i for i in range(5)]
for color, label_1, index in zip(colors, labels, indices):
    plt.plot(n_samples_array, n_neighbors_error_array[index], c= color, label = label_1, ms =3)
plt.legend(loc=2)
plt.show


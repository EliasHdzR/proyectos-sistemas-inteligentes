from sklearn.datasets import make_blobs, load_iris, load_digits, load_wine, fetch_olivetti_faces
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import RadiusNeighborsClassifier

def Fake():
    n_classes = 6
    data, labels = make_blobs(n_samples=1000,
                              centers=n_classes,
                              cluster_std = 1.3,
                              random_state=1)

    colours = ('green', 'red', 'blue', 'magenta', 'yellow', 'pink')

    fig, ax = plt.subplots()
    for n_class in range(0, n_classes):
        ax.scatter(data[labels==n_class, 0], data[labels==n_class, 1],
                   c=colours[n_class], s=10, label=str(n_class))

    #plt.show()

    from sklearn.model_selection import train_test_split

    res = train_test_split(data, labels,
                           train_size=0.7,
                           test_size=0.3,
                           random_state=1)
    train_data, test_data, train_labels, test_labels = res

    print(len(train_data), len(test_data), len(train_labels))

    X, Y = [], []
    for k in range(1, 25):
        classifier = KNeighborsClassifier(n_neighbors=k,
                                          p=2,  # Euclidian
                                          metric="minkowski")
        classifier.fit(train_data, train_labels)
        predictions = classifier.predict(test_data)
        score = accuracy_score(test_labels, predictions)
        X.append(k)
        Y.append(score)

    fig, ax = plt.subplots()
    ax.set_xlabel('k')
    ax.set_ylabel('accuracy')
    ax.plot(X, Y, "g-.o")

def digits():
    digits = load_digits()
    data = digits.data
    labels = digits.target

    res = train_test_split(data, labels,
                              train_size=0.7,
                              test_size=0.3,
                              random_state=1)

    fig, ax = plt.subplots()
    train_data, test_data, train_labels, test_labels = res
    X, Y = [], []
    for k in range(1, 25):
        classifier = KNeighborsClassifier(n_neighbors=k,
                                          p=2,  # Euclidian
                                          metric="minkowski")
        classifier.fit(train_data, train_labels)
        predictions = classifier.predict(test_data)
        score = accuracy_score(test_labels, predictions)
        X.append(k)
        Y.append(score)

    ax.set_xlabel('k')
    ax.set_ylabel('accuracy')
    ax.plot(X, Y, "g-.o")
    mejor_k = X[Y.index(max(Y))]
    print("mejor k: ", mejor_k, "-", max(Y))


def iris():
    digits = load_iris()
    data = digits.data
    labels = digits.target

    res = train_test_split(data, labels,
                              train_size=0.7,
                              test_size=0.3,
                              random_state=1)

    fig, ax = plt.subplots()
    train_data, test_data, train_labels, test_labels = res
    X, Y = [], []
    for k in range(1, 25):
        classifier = KNeighborsClassifier(n_neighbors=k,
                                          p=2,  # Euclidian
                                          metric="minkowski")
        classifier.fit(train_data, train_labels)
        predictions = classifier.predict(test_data)
        score = accuracy_score(test_labels, predictions)
        X.append(k)
        Y.append(score)

    ax.set_xlabel('k')
    ax.set_ylabel('accuracy')
    ax.plot(X, Y, "g-.o")
    mejor_k = X[Y.index(max(Y))]
    print("mejor k: ", mejor_k, "-", max(Y))

def wine():
    digits = load_wine()
    data = digits.data
    labels = digits.target

    res = train_test_split(data, labels,
                           train_size=0.7,
                           test_size=0.3,
                           random_state=1)

    fig, ax = plt.subplots()
    train_data, test_data, train_labels, test_labels = res
    X, Y = [], []
    for k in range(1, 25):
        classifier = KNeighborsClassifier(n_neighbors=k,
                                          p=2,  # Euclidian
                                          metric="minkowski")
        classifier.fit(train_data, train_labels)
        predictions = classifier.predict(test_data)
        score = accuracy_score(test_labels, predictions)
        X.append(k)
        Y.append(score)

    ax.set_xlabel('k')
    ax.set_ylabel('accuracy')
    ax.plot(X, Y, "g-.o")
    mejor_k = X[Y.index(max(Y))]
    print("mejor k: ", mejor_k, "-", max(Y))

def faces():
    digits = fetch_olivetti_faces()
    data = digits.data
    labels = digits.target

    res = train_test_split(data, labels,
                           train_size=0.7,
                           test_size=0.3,
                           random_state=1)

    fig, ax = plt.subplots()
    train_data, test_data, train_labels, test_labels = res
    X, Y = [], []
    for k in range(1, 25):
        classifier = KNeighborsClassifier(n_neighbors=k,
                                          p=2,  # Euclidian
                                          metric="minkowski")
        classifier.fit(train_data, train_labels)
        predictions = classifier.predict(test_data)
        score = accuracy_score(test_labels, predictions)
        X.append(k)
        Y.append(score)

    ax.set_xlabel('k')
    ax.set_ylabel('accuracy')
    ax.plot(X, Y, "g-.o")
    mejor_k = X[Y.index(max(Y))]
    print("mejor k: ", mejor_k, "-", max(Y))

def radiusIris():
    digits = load_wine()
    data = digits.data
    labels = digits.target

    res = train_test_split(data, labels,
                           train_size=0.7,
                           test_size=0.3,
                           random_state=1)

    fig, ax = plt.subplots()
    train_data, test_data, train_labels, test_labels = res
    X, Y = [], []
    Radio = 0.1

    while True:
        try:
            classifier = RadiusNeighborsClassifier(radius=Radio)
            classifier.fit(train_data, train_labels)
            classifier.predict(test_data)
        except ValueError:
            Radio+=0.1
        else:
            break

    print("menor radio: ", Radio)

    for k in range(1, 10):
        classifier = RadiusNeighborsClassifier(radius=Radio)
        classifier.fit(train_data, train_labels)
        predictions = classifier.predict(test_data)
        score = accuracy_score(test_labels, predictions)
        #X.append(k)
        X.append(Radio)
        Y.append(score)
        print(score)
        Radio += 0.5

    ax.set_xlabel('k')
    ax.set_ylabel('accuracy')
    ax.plot(X, Y, "g-.o")
    mejor_k = X[Y.index(max(Y))]
    print("mejor k: ", mejor_k, "-", max(Y))

#digits()
#iris()
#wine()
#faces()
radiusIris()
plt.show()
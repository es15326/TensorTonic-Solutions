import numpy as np

def majority_classifier(y_train, X_test):
    """
    Predict the most frequent label in training data for all test samples.
    """
    # Write code here
    y_train = np.array(y_train)
    X_test = np.array(X_test)

    classes, counts = np.unique(y_train, return_counts=True)
    max_count = np.max(counts)
    top_classes = classes[counts == max_count]

    first_occurence = [np.where(y_train == c)[0][0] for c in top_classes]
    majority_class = top_classes[np.argmin(first_occurence)]

    preds = np.full((X_test.shape[0], ), majority_class)

    return preds
    
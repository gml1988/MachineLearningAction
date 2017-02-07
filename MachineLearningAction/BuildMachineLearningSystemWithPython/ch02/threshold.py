import numpy as np

def learn_model(features, labels):
    best_acc = -1.0
    # Loop over all the features
    for fi in range(features.shape[1]):
        thresh = features[:, fi].copy()
        thresh.sort()
        for t in thresh:
            pred = (features[:, fi] > t)

            acc = (pred == labels).mean()
            rev_acc = (pred == ~labels).mean()

            if rev_acc > acc:
                acc = rev_acc
                reverse = True
            else:
                reverse = False

            if acc > best_acc:
                best_acc = acc
                best_fi = fi
                best_th = t
                best_reverse = reverse

    return best_th, best_fi, best_reverse

def apply_model(model, features):
    th, fi, reverse = model
    if reverse:
        return features[:, fi] <= th
    else:
        return features[:, fi] > th

def accuracy(features, labels, model):
    pred = predict(model, features)
    return np.mean(pred == labels)
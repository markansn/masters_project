def remove_class(X, class_name):
    for i, apk in enumerate(X):
        features_to_remove = []
        for feature in apk:
            if class_name + "::" in feature:
                features_to_remove.append(feature)
        for f in features_to_remove:
            del X[i][f]

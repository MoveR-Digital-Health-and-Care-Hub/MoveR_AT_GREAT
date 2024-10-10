from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score


def lda_acc(X_train, y_train, X_test, y_test, solver="svd"):
    if solver == "svd":
        lda = LinearDiscriminantAnalysis(store_covariance=True)
    elif solver == "lsqr":
        lda = LinearDiscriminantAnalysis(solver="lsqr")
    elif solver == "eigen":
        lda = LinearDiscriminantAnalysis(solver="eigen")

    # Train the LDA classifier
    lda.fit(X_train, y_train)
    mean_ = lda.means_
    cov_ = lda.covariance_

    train_acc = accuracy_score(y_train, lda.predict(X_train))
    test_pred = lda.predict(X_test)
    test_acc = accuracy_score(y_test, test_pred)
    return test_acc, train_acc #  mean_, cov_

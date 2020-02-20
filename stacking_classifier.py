import pandas as pd


class StackingClassifier:
    """A 'Stacking Cross-Validation' classifier.

    Parameters
    ----------
    base_classifiers : array-like, shape = [n_classifiers]
        A list of classifiers to use for the first layer of the ensemble.
    meta_classifier : object
        The meta-classifier to be fitted on the ensemble of
        classifiers
    cv_splitter : int, cross-validation generator
        Determines the cross-validation splitting strategy.
    Attributes
    ----------
    _base_classifiers : list, shape=[n_classifiers]
        Fitted classifiers
    _meta_classifier : estimator
        Fitted meta-classifier
    """

    def __init__(self, base_classifiers=None, meta_classifier=None, cv_splitter=None):
        self._base_classifiers = base_classifiers if base_classifiers is not None else []
        self._meta_classifier = meta_classifier
        self._cv_splitter = cv_splitter

    @property
    def base_learners(self):
        """Get the first layer of classifiers."""
        return self._base_classifiers

    def add_base_learner(self, base_learner):
        """Add a classifier to the first layer."""
        self._base_classifiers.append(base_learner)

    @property
    def meta_learner(self):
        """Get the classifier for the end layer."""
        return self._meta_classifier

    @meta_learner.setter
    def meta_learner(self, value):
        """Set a classifier for the end layer."""
        self._meta_classifier = value

    def fit(self, X, y):
        """Fit the ensemble according to the given training data.

        Parameters
        ----------
        X : pd.DataFrame, shape = [n_samples, n_features]
            Training vector, where n_samples is the number of samples and
            n_features is the number of features.

        y : pd.Series, shape = [n_samples,]
            Target vector relative to X.

        Returns
        -------
        self : object
        """
        self._fit_meta_learner(X, y)
        self._fit_base_learners(X, y)
        return self

    def predict(self, X):
        """Perform classification on samples in X.

        Parameters
        ----------
        X : pd.DataFrame(), shape = [n_samples, n_features]
            Training vector, where n_samples is the number of samples and
            n_features is the number of features.

        Returns
        -------
        pred_classes : pd.Series, shape = [n_samples]
            Returns class labels for samples in X.
        pred_proba : pd.Series, shape = [n_samples]
            Returns the probability of the sample for each class in the model.
        """
        y_pred_base = self._predict_with_base_learners(X)
        pred_proba = self._predict_with_meta_learner(y_pred_base)
        pred_classes = 1 * (pred_proba >= .5)
        return pred_classes, pred_proba

    def _fit_base_learners(self, X, y):
        for base_learner in self._base_classifiers:
            base_learner.fit(X, y)

    def _fit_meta_learner(self, X, y):
        y_pred = pd.DataFrame()
        y_true = pd.Series()
        for train, validation in self._split_data(X, y):
            X_train, y_train = X.iloc[train], y.iloc[train]
            X_val, y_val = X.iloc[validation], y.iloc[validation]
            self._fit_base_learners(X_train, y_train)
            prediction = self._predict_with_base_learners(X_val)
            y_pred = y_pred.append(prediction, ignore_index=True, sort=False)#.dropna(axis=1)
            y_true = y_true.append(y_val, ignore_index=True)
        self._meta_classifier.fit(X=y_pred, y=y_true)

    def _predict_with_base_learners(self, X):
        y_pred_proba_all_base_learners = []
        for base_learner in self._base_classifiers:
            y_pred_proba_single_base_learner = base_learner.predict_proba(X)[:, 1]
            y_pred_proba_all_base_learners += [y_pred_proba_single_base_learner]
        y_pred_proba_all_base_learners = pd.DataFrame(y_pred_proba_all_base_learners).transpose()
        return y_pred_proba_all_base_learners

    def _predict_with_meta_learner(self, X):
        X = X.fillna(.5)
        y_pred = self._meta_classifier.predict_proba(X)[:, 1]
        return y_pred

    def _split_data(self, X, y):
        groups = self._calc_groups(X)
        return self._cv_splitter.split(X, y, groups=groups)

    @staticmethod
    def _calc_groups(data):
        return pd.Categorical(data['structure_id']).codes

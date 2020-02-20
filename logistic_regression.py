import numpy as np
from sklearn.linear_model.logistic import LogisticRegression as LR


class LogisticRegression:
    """Logistic Regression classifier.

    It encapsulates Sklearn's Logistic Regression algorithm,
    and provides an interface to load a model from parameters for prediction purposes.

    Parameters
    ----------
    regularization_strength : float, optional (default=1e3)
        Bigger values specify stronger regularization.
    features_subset : list, optional (default=None)
        A subset of features the model has to take account of.
    class_weight : dict or 'balanced', optional (default=None)
        Weights associated with classes in the form ``{class_label: weight}``.
        If not given, all classes are supposed to have weight one.
        The "balanced" mode uses the values of y to automatically adjust
        weights inversely proportional to class frequencies in the input data
        as ``n_samples / (n_classes * np.bincount(y))``.
        Note that these weights will be multiplied with sample_weight (passed
        through the fit method) if sample_weight is specified.
    random_state : int, optional (default=None)
        Seed for the random number generator.

    Attributes
    ----------
    _model
        Logistic regression model
    """

    def __init__(self, regularization_strength=1e3, features_subset=None, class_weight=None, random_state=None):
        self._model = LR(C=1/regularization_strength, solver='lbfgs', class_weight=class_weight,
                         random_state=random_state)
        self._features_subset = features_subset

    def fit(self, X, y):
        """Fit the model according to the given training data.

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
        if self._features_subset is None:
            columns = X.columns
        else:
            columns = self._features_subset
        mask_not_na = (~X[columns].isna()).all(axis=1)
        self._model.fit(X.loc[mask_not_na, columns], y[mask_not_na])
        return self

    def predict_proba(self, X):
        """Probability estimates.

        Parameters
        ----------
        X : pd.DataFrame(), shape = [n_samples, n_features]
            Training vector, where n_samples is the number of samples and
            n_features is the number of features.

        Returns
        -------
        array-like, shape = [n_samples, n_classes]
            Returns the probability of the sample for each class in the model.
        """
        if self._features_subset is None:
            columns = X.columns
        else:
            columns = self._features_subset
        mask_not_na_test = (~X[columns].isna()).all(axis=1)
        return self._model.predict_proba(X.loc[mask_not_na_test, columns])

    @property
    def intercept(self):
        """Get the intercept added to the decision function."""
        return self._model.intercept_[0]

    @intercept.setter
    def intercept(self, value):
        """Set the intercept to add to decision function."""
        self._model.intercept_ = np.array(value)
        return self

    @property
    def coefficients(self):
        """Get the coefficient of the features in the decision function."""
        return self._model.coef_[0]

    def get_coefficient(self, i):
        """Get the coefficient of the i-th feature in the decision function.

        Parameters
        ----------
        i : int
            index of a coefficient in the decision function.
        """
        return self.coefficients[i]

    @coefficients.setter
    def coefficients(self, value):
        """Set the coefficient of each feature in the decision function."""
        self._model.coef_ = np.array([value])
        return self

    def add_coefficient(self, coefficient):
        """Add the coefficient of a features in the decision function."""
        if not hasattr(self._model, "coef_"):
            self._model.coef_ = np.array([[]])
        self._model.coef_ = np.hstack((self._model.coef_, np.array([[coefficient]])))
        return self

    def get_feature_name(self, i):
        """Get the name of the i-th feature in the decision function.

        Parameters
        ----------
        i : int
            index of a feature in the decision function.
        """
        return self._features_subset[i]

    def set_classes(self, classes, dtype):
        """Define a mapping for each class in the model.

        Parameters
        ----------
        classes : array-like, shape = [n_classes,]
            target classes
        dtype : dtype : str or dtype
            Typecode or data-type of the classes.
        """
        self._model.classes_ = np.array(classes, dtype)
        return self

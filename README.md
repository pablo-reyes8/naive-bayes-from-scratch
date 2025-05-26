# Pure Python Categorical Naive Bayes

A from-scratch implementation of a Categorical Naive Bayes classifier in pure Python, following a scikit-learn–style API. This package supports Laplace–smoothed estimation of class priors and conditional likelihoods for categorical features, and delivers identical predictions to `sklearn.naive_bayes.CategoricalNB`.

---

## 🚀 Features

- **Fit / Predict API**  
  - `fit(X, y, alpha=1)` estimates class priors and feature likelihoods with optional Laplace smoothing  
  - `predict(x_new)` returns the predicted class and log-posterior scores  
- **Exact Match with scikit-learn**  
  Validated to produce identical predictions to `CategoricalNB` on representative datasets  
- **Extensible Design**  
  Easily subclass or extend for Gaussian NB, mixed‐type data, or custom smoothing strategies  
- **Lightweight**  
  Only depends on Pandas and NumPy (and scikit-learn for demonstration/comparison)

---

## 📦 API Reference

### `NaiveBayes(alpha=1.0)`

- **Parameters**  
  - `alpha` (_float_, default=1.0): Laplace smoothing parameter

- **Attributes after `fit`**  
  - `.prior` (_dict_): `{class_label: P(Y=class_label)}`  
  - `.likelihood` (_dict_): `{class_label: {feature_name: {value: P(value|class)}}}`  
  - `.labels` (_list_): Feature column names  
  - `.cols` (_int_): Number of features

### `fit(X, y, alpha=None, parameters=False)`

- **Parameters**  
  - `X` (_pd.DataFrame_): Categorical feature matrix  
  - `y` (_pd.Series_): Target labels  
  - `alpha` (_float_, optional): Override smoothing parameter  
  - `parameters` (_bool_): If `True`, returns `(likelihood, prior, class_counts)`  

- **Returns**  
  - Nothing (unless `parameters=True`)

### `predict(x_new)`

- **Parameters**  
  - `x_new` (_list_): Single observation feature values  

- **Returns**  
  - Tuple `(predicted_class, log_posterior_dict)`

---

## 🛠️ Next Steps

1. **Gaussian NB** – Extend to continuous features with Gaussian likelihoods  
2. **Mixed Data** – Combine categorical and numerical support in one class  
3. **Pipeline Integration** – Subclass `BaseEstimator` and `ClassifierMixin` for full scikit-learn compatibility  
4. **Performance** – Vectorize loops or add Cython for speed, handle large datasets

---
## Contributing

Contributions are welcome! Please open issues or submit pull requests at  
https://github.com/pablo-reyes8


## 📄 License

MIT License © 2025  

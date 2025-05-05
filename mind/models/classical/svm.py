import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import RandomizedSearchCV
from sklearn.decomposition import PCA
from typing import Dict, Any, Tuple, List, Optional
import logging

logger = logging.getLogger(__name__)


def create_svm(config: Dict[str, Any], signal_type: str = None) -> SVC:
    """
    Create a Support Vector Machine classifier with the specified configuration.

    Parameters
    ----------
    config : Dict[str, Any]
        Configuration dictionary
    signal_type : str, optional
        Signal type to optimize for, by default None

    Returns
    -------
    SVC
        Configured SVM model
    """
    svm_params = config['models']['classical']['svm'].copy()

    # Enhanced parameters for deconvolved signals
    if signal_type == 'deconv':
        svm_params.update({
            'kernel': 'rbf',
            'C': 10.0,  # Higher C for deconvolved signals
            'gamma': 'scale',
            'class_weight': 'balanced',
            'probability': True,
            'cache_size': 1000,  # Larger cache for faster training
            'tol': 1e-4  # Lower tolerance for better optimization
        })
    elif signal_type == 'deltaf':
        # Moderate parameters for deltaf signals
        svm_params.update({
            'kernel': 'rbf',
            'C': 5.0,
            'gamma': 'scale',
            'probability': True
        })
    # Default parameters for calcium signals remain unchanged

    model = SVC(
        kernel=svm_params.get('kernel', 'rbf'),
        C=svm_params.get('C', 1.0),
        gamma=svm_params.get('gamma', 'scale'),
        class_weight=svm_params.get('class_weight', 'balanced'),
        probability=svm_params.get('probability', True),
        cache_size=svm_params.get('cache_size', 200),
        tol=svm_params.get('tol', 1e-3),
        random_state=config['experiment'].get('seed', 42)
    )

    return model


def optimize_svm(
        X_train: np.ndarray,
        y_train: np.ndarray,
        config: Dict[str, Any],
        class_weights: Optional[Dict[int, float]] = None,
        signal_type: str = None
) -> Tuple[SVC, Dict[str, Any], Optional[PCA]]:
    """
    Optimize SVM hyperparameters using randomized search, with efficient PCA.

    Parameters
    ----------
    X_train : np.ndarray
        Training data
    y_train : np.ndarray
        Training labels
    config : Dict[str, Any]
        Configuration dictionary
    class_weights : Optional[Dict[int, float]], optional
        Class weights, by default None
    signal_type : str, optional
        Signal type to optimize for, by default None

    Returns
    -------
    Tuple[SVC, Dict[str, Any], Optional[PCA]]
        Optimized SVM model, best parameters, and PCA transformer (if used)
    """
    logger.info(f"Optimizing SVM hyperparameters for {signal_type if signal_type else 'general'} data")

    # Check if PCA should be applied
    svm_params = config['models']['classical']['svm']
    use_pca = svm_params.get('pca', True)
    pca_transformer = None

    # Apply PCA if requested
    if use_pca:
        n_features = X_train.shape[1]

        # Determine optimal number of components
        if signal_type == 'deconv':
            # Use higher explained variance for deconvolved signals
            pca_components = 0.98
        else:
            pca_components = svm_params.get('pca_components', 0.95)

        if isinstance(pca_components, float) and pca_components <= 1.0:
            # Use explained variance ratio
            n_components = pca_components
        else:
            # Use specific number of components (capped at 100 for efficiency)
            n_components = min(n_features, int(pca_components), 100)

        logger.info(f"Applying PCA to reduce dimensions from {n_features} to {n_components}")
        pca_transformer = PCA(n_components=n_components, random_state=config['experiment'].get('seed', 42))
        X_train_pca = pca_transformer.fit_transform(X_train)

        # Log explained variance
        explained_variance = np.sum(pca_transformer.explained_variance_ratio_)
        logger.info(f"PCA explained variance ratio: {explained_variance:.4f}")

        # Use transformed data for hyperparameter optimization
        X_train_opt = X_train_pca
    else:
        X_train_opt = X_train

    # Define parameter distribution for randomized search
    param_dist = {
        'C': [0.1, 1.0, 10.0, 100.0],
        'gamma': ['scale', 'auto', 0.01, 0.1, 1.0],
        'kernel': ['rbf', 'linear', 'poly'],
        'probability': [True]
    }

    # Modify search space based on signal type
    if signal_type == 'deconv':
        # Biased search space for deconvolved signals
        param_dist.update({
            'C': [5.0, 10.0, 50.0, 100.0],  # Higher C values
            'gamma': ['scale', 0.1, 0.5, 1.0],  # More complex kernels
            'kernel': ['rbf', 'poly'],  # Prefer nonlinear kernels
            'degree': [2, 3]  # For poly kernel
        })

    # If class_weights provided, include them
    if class_weights:
        param_dist['class_weight'] = ['balanced', class_weights]
    else:
        param_dist['class_weight'] = ['balanced']

    # Initialize SVM
    svm = SVC(random_state=config['experiment'].get('seed', 42))

    # Randomized search with cross-validation using efficient parameters
    random_search = RandomizedSearchCV(
        estimator=svm,
        param_distributions=param_dist,
        n_iter=10,  # Reduced number of parameter settings
        cv=3,
        n_jobs=-1,
        scoring='f1_weighted',
        random_state=config['experiment'].get('seed', 42),
        verbose=0
    )

    # Fit model
    logger.info(f"Performing randomized search for SVM with {signal_type if signal_type else 'general'} data")
    random_search.fit(X_train_opt, y_train.astype(int))

    # Get best parameters
    best_params = random_search.best_params_
    logger.info(f"Best parameters for {signal_type if signal_type else 'general'}: {best_params}")

    return random_search.best_estimator_, best_params, pca_transformer

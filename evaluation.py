
# evaluation.py - Model Evaluation and Visualization
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, log_loss, confusion_matrix, classification_report, roc_auc_score, roc_curve
import numpy as np
from sklearn.decomposition import PCA
from scipy.spatial.distance import jensenshannon

def evaluate_model(X_train, X_test, y_train, y_test):
    # Load the best model and vectorizer
    import pickle
    with open('best_model.pkl', 'rb') as model_file:
        best_model = pickle.load(model_file)
    with open('vectorizer.pkl', 'rb') as vectorizer_file:
        vectorizer = pickle.load(vectorizer_file)

    # Predict on the test set
    y_pred = best_model.predict(X_test)
    y_pred_proba = best_model.predict_proba(X_test)[:, 1]

    # Calculate accuracy and log loss
    accuracy = accuracy_score(y_test, y_pred)
    loss = log_loss(y_test, y_pred_proba)
    print(f'Accuracy: {accuracy:.2f}')
    print(f'Log Loss: {loss:.2f}')

    # Confusion Matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(conf_matrix)

    # Classification Report
    class_report = classification_report(y_test, y_pred)
    print("Classification Report:")
    print(class_report)

    # ROC AUC Score
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    print(f'ROC AUC Score: {roc_auc:.2f}')

    # Plotting ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, color='b', label=f'ROC Curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='r', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.show()

    # Data Drift Monitoring using Jensen-Shannon Divergence
    train_features_mean = np.mean(X_train.toarray(), axis=0)
    test_features_mean = np.mean(X_test.toarray(), axis=0)

    # Calculate the Jensen-Shannon Divergence between training and test distributions
    js_divergence = jensenshannon(train_features_mean, test_features_mean)
    print(f'Jensen-Shannon Divergence between training and test feature distributions: {js_divergence:.4f}')

    # Alert if JS divergence is above a reasonable threshold indicating potential data drift
    js_threshold = 0.2
    if js_divergence > js_threshold:
        print("WARNING: Significant data drift detected! Jensen-Shannon Divergence is above the threshold.")

    # PCA for Visualizing Clusters
    pca = PCA(n_components=4)
    X_train_pca = pca.fit_transform(X_train.toarray())
    X_test_pca = pca.transform(X_test.toarray())

    # Plot PCA clusters in 3D (first 3 components)
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X_train_pca[:, 0], X_train_pca[:, 1], X_train_pca[:, 2], c=y_train, cmap='viridis', alpha=0.5, label='Training Data')
    ax.scatter(X_test_pca[:, 0], X_test_pca[:, 1], X_test_pca[:, 2], c='red', alpha=0.5, label='Test Data')
    ax.set_xlabel('PCA Component 1')
    ax.set_ylabel('PCA Component 2')
    ax.set_zlabel('PCA Component 3')
    ax.set_title('PCA Visualization of Training and Test Data (First 3 Components)')
    ax.legend()
    plt.show()

    # Plotting all 4 PCA components in pairs
    fig, axs = plt.subplots(2, 3, figsize=(18, 12))
    components = ['PCA Component 1', 'PCA Component 2', 'PCA Component 3', 'PCA Component 4']
    combinations = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]

    for ax, (i, j) in zip(axs.ravel(), combinations):
        ax.scatter(X_train_pca[:, i], X_train_pca[:, j], c=y_train, cmap='viridis', alpha=0.5, label='Training Data')
        ax.scatter(X_test_pca[:, i], X_test_pca[:, j], c='red', alpha=0.5, label='Test Data')
        ax.set_xlabel(components[i])
        ax.set_ylabel(components[j])
        ax.set_title(f'{components[i]} vs. {components[j]}')
        ax.legend()

    plt.tight_layout()
    plt.show()

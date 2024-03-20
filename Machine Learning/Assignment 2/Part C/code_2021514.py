import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

class MyDecisionTree:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth

    def fit(self, X, y):
        self.X = X
        self.y = y
        self.tree = self._build_tree(X, y, depth=0)

    def _build_tree(self, X, y, depth):
        if depth == self.max_depth or len(set(y)) == 1:
            return np.bincount(y).argmax()

        best_split = self._find_best_split(X, y)
        if best_split is None:
            return np.bincount(y).argmax()

        feature, threshold, left, right = best_split
        left_subtree = self._build_tree(X[left], y[left], depth + 1)
        right_subtree = self._build_tree(X[right], y[right], depth + 1)

        return (feature, threshold, left_subtree, right_subtree)

    def _find_best_split(self, X, y):
        num_features = X.shape[1]
        best_gini = float('inf')
        best_split = None

        for feature in range(num_features):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                left_indices = X[:, feature] < threshold
                # print(left_indices)
                right_indices = X[:, feature] >= threshold

                if len(left_indices) == 0 or len(right_indices) == 0:
                    continue

                gini = self._compute_gini_impurity(y, left_indices, right_indices)

                if gini < best_gini:
                    best_gini = gini
                    best_split = (feature, threshold, left_indices, right_indices)
    
        return best_split

    def _compute_gini_impurity(self, y, left_indices, right_indices):
        left_gini = self._gini_index(y[left_indices])
        right_gini = self._gini_index(y[right_indices])

        total_samples = len(y)
        p_left = len(y[left_indices]) / total_samples
        p_right = len(y[right_indices]) / total_samples

        gini_impurity = p_left * left_gini + p_right * right_gini
        return gini_impurity

    def _gini_index(self, y):
        if len(y) == 0:
            return 0
        p = np.bincount(y) / len(y)
        return 1 - np.sum(p ** 2)

    def predict(self, X):
        return np.array([self._predict_one(x, self.tree) for x in X])

    def _predict_one(self, x, node):
        if isinstance(node, int):
            return node
        if isinstance(node, np.int64):
            return int(node)
        feature, threshold, left, right = node

        if x[feature] < threshold:
            return self._predict_one(x, left)
        else:
            return self._predict_one(x, right)

    def score(self, X, y):
        y_pred = self.predict(X)
        return np.mean(y_pred == y)

# Load the encoded data
data = pd.read_csv('encoded_data.csv')
labels = data.columns[-1]
data[labels] = data[labels].apply(lambda x: 0 if x==3 else 1)
# Split the data into training and testing sets
X = data.drop(columns=["label"]).to_numpy()
y = data["label"].to_numpy()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  # Adjust the test_size and random_state as needed


tree = MyDecisionTree(max_depth=20)
tree.fit(X_train, y_train)

# Calculate accuracy on the training data
train_accuracy = tree.score(X_train, y_train)
print("Training Accuracy:", train_accuracy)

# Calculate accuracy on the testing data
test_accuracy = tree.score(X_test, y_test)
print("Testing Accuracy:", test_accuracy)
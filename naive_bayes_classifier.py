import re
import os
import random
import tarfile
from collections import defaultdict
from math import log
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt
import numpy as np

class NVclass:
    def __init__(self, alpha: float = 1.0):
        # Laplace smoothing and class-level data
        self.alpha = alpha
        self.c_label = []       # Class labels
        self.c_prior = {}       # P(class)
        self.w_count = {}       # Word counts per class
        self.cw_total = {}      # Total words per class
        self.unique = set()     # Vocabulary set

    def preprocess(self, text: str) -> List[str]:
        """Lowercase, remove punctuation, and tokenize."""
        text = text.lower()
        text = re.sub(r'[^a-z0-9\s]', ' ', text)
        tokens = text.split()
        tokens = [t for t in tokens if len(t) > 1]
        return tokens

    def train(self, docs: List[str], labels: List[str]):
        """Train Naive Bayes classifier."""
        self.c_label = sorted(list(set(labels)))
        n_docs = len(docs)

        # Initialize
        doc_count = defaultdict(int)
        self.w_count = {cls: defaultdict(int) for cls in self.c_label}
        self.cw_total = {cls: 0 for cls in self.c_label}

        # Count words and documents
        for document, label in zip(docs, labels):
            doc_count[label] += 1
            tokens = self.preprocess(document)
            for token in tokens:
                self.unique.add(token)
                self.w_count[label][token] += 1
                self.cw_total[label] += 1

        # Calculate class prior probabilities (log scale)
        for cls in self.c_label:
            self.c_prior[cls] = log(doc_count[cls] / n_docs)

    def calc_log(self, word: str, cls: str) -> float:
        """Compute log probability of word given class."""
        word_count = self.w_count[cls].get(word, 0)
        total_words = self.cw_total[cls]
        vocab_size = len(self.unique)

        prob = (word_count + self.alpha) / (total_words + self.alpha * vocab_size)
        return log(prob)

    def predict_single(self, document: str) -> str:
        """Predict class for a single document."""
        tokens = self.preprocess(document)
        class_scores = {}

        for cls in self.c_label:
            score = self.c_prior[cls]
            for token in tokens:
                if token in self.unique:
                    score += self.calc_log(token, cls)
            class_scores[cls] = score

        return max(class_scores, key=class_scores.get)

    def predict(self, docs: List[str]) -> List[str]:
        """Predict classes for multiple documents."""
        return [self.predict_single(doc) for doc in docs]

    def evaluate(self, docs: List[str], true_labels: List[str]) -> Dict:
        """Evaluate model performance on test data."""
        predict = self.predict(docs)

        # Accuracy
        correct = sum(1 for pred, true in zip(predict, true_labels) if pred == true)
        accuracy = correct / len(true_labels)

        # Class metrics
        class_metrics = {}
        for cls in self.c_label:
            tp = sum(1 for p, t in zip(predict, true_labels) if p == cls and t == cls)
            fp = sum(1 for p, t in zip(predict, true_labels) if p == cls and t != cls)
            fn = sum(1 for p, t in zip(predict, true_labels) if p != cls and t == cls)

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = (2 * precision * recall / (precision + recall)
                  if (precision + recall) > 0 else 0)

            class_metrics[cls] = {'precision': precision, 'recall': recall, 'f1': f1}

        # Macro averages
        m_precision = sum(m['precision'] for m in class_metrics.values()) / len(self.c_label)
        m_recall = sum(m['recall'] for m in class_metrics.values()) / len(self.c_label)
        macro_f1 = sum(m['f1'] for m in class_metrics.values()) / len(self.c_label)

        # Confusion matrix
        confusion_matrix = {true_cls: {pred_cls: 0 for pred_cls in self.c_label}
                            for true_cls in self.c_label}
        for pred, true in zip(predict, true_labels):
            confusion_matrix[true][pred] += 1

        return {
            'accuracy': accuracy,
            'm_precision': m_precision,
            'm_recall': m_recall,
            'macro_f1': macro_f1,
            'class_metrics': class_metrics,
            'confusion_matrix': confusion_matrix
        }


def load_data(data_dir: str) -> Tuple[List[str], List[str]]:
    """Load dataset from directory structure."""
    docs = []
    labels = []

    # Get all newsgroup directories
    newsgroups = sorted([
        d for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d))
    ])

    # Load documents from each newsgroup
    for newsgroup in newsgroups:
        newsgroup_path = os.path.join(data_dir, newsgroup)
        files = [
            f for f in os.listdir(newsgroup_path)
            if os.path.isfile(os.path.join(newsgroup_path, f))
        ]

        for filename in files:
            filepath = os.path.join(newsgroup_path, filename)
            try:
                with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                    docs.append(content)
                    labels.append(newsgroup)
            except Exception:
                pass

    return docs, labels


def split_dataset(docs: List[str], labels: List[str],
                  train_ratio: float = 0.5, seed: int = 14) -> Tuple:
    """Split dataset into training and test sets."""
    random.seed(seed)
    indices = list(range(len(docs)))
    random.shuffle(indices)

    split_point = int(len(docs) * train_ratio)
    train_idx, test_idx = indices[:split_point], indices[split_point:]

    train_docs = [docs[i] for i in train_idx]
    train_labels = [labels[i] for i in train_idx]
    test_docs = [docs[i] for i in test_idx]
    test_labels = [labels[i] for i in test_idx]

    return train_docs, train_labels, test_docs, test_labels

def eval_results(results: Dict):
    print(f"\nOverall Accuracy: {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)")
    print(f"\nAverage Metrics:")
    print(f" Precision: {results['m_precision']:.4f}")
    print(f" Recall:    {results['m_recall']:.4f}")
    print(f" F1-Score:  {results['macro_f1']:.4f}")

    print(f"\n{'Class':<30} {'Precision':>10} {'Recall':>10} {'F1':>10}")

    for cls, metrics in sorted(results['class_metrics'].items()):
        print(f"{cls:<30} {metrics['precision']:>10.4f} {metrics['recall']:>10.4f} "
              f"{metrics['f1']:>10.4f}")

    # save confusion matrix as image
    classes = sorted(results['class_metrics'].keys())
    cm = results['confusion_matrix']

    # convert to numpy array
    cm_array = np.array([[cm[true_cls][pred_cls] for pred_cls in classes]
                         for true_cls in classes])

    # create basic figure
    fig, ax = plt.subplots(figsize=(10, 8))

    # create table
    ax.axis('tight')
    ax.axis('off')

    table_data = []
    # header row
    header = [' '] + [cls[:15] for cls in classes]
    table_data.append(header)

    # data rows
    for i, true_cls in enumerate(classes):
        row = [true_cls[:15]] + [str(int(cm_array[i, j])) for j in range(len(classes))]
        table_data.append(row)

    table = ax.table(cellText=table_data, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.5)

    plt.savefig('confusion_matrix.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\nSee the full confusion matrix in 'confusion_matrix.png'")

def main():
    TRAIN_RATIO = 0.5
    RANDOM_SEED = 14
    DATA_DIRECTORY = "20_newsgroup"
    DATA_TAR_PATH = "20_newsgroups.tar.gz"
        
    # load the dataset
    docs, labels = load_data(DATA_DIRECTORY)

    # split data
    train_docs, train_labels, test_docs, test_labels = split_dataset(
        docs, labels, train_ratio=TRAIN_RATIO, seed=RANDOM_SEED
    )

    classifier = NVclass(alpha=1.0)
    classifier.train(train_docs, train_labels)

    # evaluate results
    results = classifier.evaluate(test_docs, test_labels)

    # print results
    eval_results(results)

if __name__ == "__main__":
    main()

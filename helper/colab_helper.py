import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def proportion_balance_classes(names: pd.Index, values: np.ndarray) -> None:
    """We plot the proportion of each class in each row."""

    plt.figure(figsize=(15, 4))
    ax = sns.barplot(x=names, y=values, alpha=0.8)
    plt.title("# per class")
    plt.ylabel('# of Occurrences', fontsize=12)
    plt.xlabel('Type', fontsize=12)

    rects = ax.patches
    for rect, label in zip(rects, values):
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width() / 2, height + 5, f'{label:.2f}', ha='center', va='bottom')

    plt.show()

    return
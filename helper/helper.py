import pandas as pd
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from typing import Tuple

from nltk.corpus import stopwords
import nltk
import string
nltk.download('stopwords')

# colorama of the charts
custom_colors = ['#36CE8A', "#7436F5","#3736F4",   "#36AEF5", "#B336F5", "#f8165e", "#36709A",  "#3672F5", "#7ACE5D"]
gradient_colors = [ "#36CE8A", '#7436F5']
color_palette_custom  = sns.set_palette(custom_colors)
theme_color = sns.color_palette(color_palette_custom, 9)
cmap_theme = LinearSegmentedColormap.from_list('custom_colormap', gradient_colors)

# color to manage in the schema
theme_color


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


def histogram_bins(names: pd.Index, values: np.ndarray, bins: int = 10, title: str = "Proportion per Bin") -> None:
    """Plot the proportion of each class in specified bins with integer value ranges on the x-axis and an optional title."""

    # Bin the values and get bin edges for creating range labels
    bin_ranges = pd.cut(values, bins=bins)
    binned_values = bin_ranges.value_counts().sort_index()

    # Generate integer range labels for x-axis based on bin edges
    bin_labels = [f"{int(interval.left)} - {int(interval.right)}" for interval in bin_ranges.categories]
    bin_counts = binned_values.values

    plt.figure(figsize=(15, 4))
    ax = sns.barplot(x=bin_labels, y=bin_counts, alpha=0.8)
    plt.title(title)  # Set the custom or default title
    plt.ylabel('# of Occurrences', fontsize=12)
    plt.xlabel('Value Ranges', fontsize=12)

    # Annotate each bar with integer counts
    rects = ax.patches
    for rect, label in zip(rects, bin_counts):
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width() / 2, height + 5, f'{int(label)}', ha='center', va='bottom')

    plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for readability
    plt.show()

    return


def extractions_text_description(dataset: pd.DataFrame, column_name: str) -> pd.DataFrame:
    """We will be adding more features to the data"""
    # import values

    # Define English stopwords
    eng_stopwords = set(stopwords.words('english'))

    # Create a copy of the original dataset
    df_eda_description = dataset.copy()

    # Word count in each comment:
    df_eda_description['count_each_word'] = df_eda_description[column_name].apply(lambda x: len(str(x).split()))
    df_eda_description['count_unique_word'] = df_eda_description[column_name].apply(lambda x: len(set(str(x).split())))
    df_eda_description['count_punctuations'] = df_eda_description[column_name].apply(
        lambda x: len([c for c in str(x) if c in string.punctuation]))
    df_eda_description['count_words_title'] = df_eda_description[column_name].apply(
        lambda x: len([w for w in str(x).split() if w.istitle()]))
    df_eda_description['count_stopwords'] = df_eda_description[column_name].apply(
        lambda x: len([w for w in str(x).lower().split() if w in eng_stopwords]))
    df_eda_description['mean_word_len'] = df_eda_description[column_name].apply(
        lambda x: np.mean([len(w) for w in str(x).split()]))

    # Additional features from the original dataset
    df_eda_description['total_length'] = df_eda_description[column_name].str.len()
    df_eda_description['new_line'] = df_eda_description[column_name].str.count('\n' * 1)
    df_eda_description['new_small_space'] = df_eda_description[column_name].str.count('\n' * 2)
    df_eda_description['new_medium_space'] = df_eda_description[column_name].str.count('\n' * 3)
    df_eda_description['new_big_space'] = df_eda_description[column_name].str.count('\n' * 4)

    # Uppercase words count
    df_eda_description['uppercase_words'] = df_eda_description[column_name].apply(
        lambda l: sum(map(str.isupper, list(l))))
    df_eda_description['question_mark'] = df_eda_description[column_name].str.count('\?')
    df_eda_description['exclamation_mark'] = df_eda_description[column_name].str.count('!')

    # Derived features
    df_eda_description['word_unique_percent'] = df_eda_description['count_unique_word'] * 100 / df_eda_description[
        'count_each_word']
    df_eda_description['punctuations_percent'] = df_eda_description['count_punctuations'] * 100 / df_eda_description[
        'count_each_word']

    return df_eda_description
def extractions_text_description(dataset: pd.DataFrame, column_name: str) -> pd.DataFrame:
    """We will be adding more features to the data"""
    # import values

    # Define English stopwords
    eng_stopwords = set(stopwords.words('english'))

    # Create a copy of the original dataset
    df_eda_description = dataset.copy()

    # Word count in each comment:
    df_eda_description['count_each_word'] = df_eda_description[column_name].apply(lambda x: len(str(x).split()))
    df_eda_description['count_unique_word'] = df_eda_description[column_name].apply(lambda x: len(set(str(x).split())))
    df_eda_description['count_punctuations'] = df_eda_description[column_name].apply(
        lambda x: len([c for c in str(x) if c in string.punctuation]))
    df_eda_description['count_words_title'] = df_eda_description[column_name].apply(
        lambda x: len([w for w in str(x).split() if w.istitle()]))
    df_eda_description['count_stopwords'] = df_eda_description[column_name].apply(
        lambda x: len([w for w in str(x).lower().split() if w in eng_stopwords]))
    df_eda_description['mean_word_len'] = df_eda_description[column_name].apply(
        lambda x: np.mean([len(w) for w in str(x).split()]))

    # Additional features from the original dataset
    df_eda_description['total_length'] = df_eda_description[column_name].str.len()
    df_eda_description['new_line'] = df_eda_description[column_name].str.count('\n' * 1)
    df_eda_description['new_small_space'] = df_eda_description[column_name].str.count('\n' * 2)
    df_eda_description['new_medium_space'] = df_eda_description[column_name].str.count('\n' * 3)
    df_eda_description['new_big_space'] = df_eda_description[column_name].str.count('\n' * 4)

    # Uppercase words count
    df_eda_description['uppercase_words'] = df_eda_description[column_name].apply(
        lambda l: sum(map(str.isupper, list(l))))
    df_eda_description['question_mark'] = df_eda_description[column_name].str.count('\?')
    df_eda_description['exclamation_mark'] = df_eda_description[column_name].str.count('!')

    # Derived features
    df_eda_description['word_unique_percent'] = df_eda_description['count_unique_word'] * 100 / df_eda_description[
        'count_each_word']
    df_eda_description['punctuations_percent'] = df_eda_description['count_punctuations'] * 100 / df_eda_description[
        'count_each_word']

    return df_eda_description

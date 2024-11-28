# Toxic comment algorithm

![Toxic comments platform](img/cat_image.jpeg)

## Project Description and Objectives

The goal of this project is to develop an automated system for detecting toxic comments across various social media platforms. This project is designed to address a common challenge faced by data consultants: efficiently managing and moderating large volumes of user-generated content.

The primary objective is to reduce the workload of agents who manually review comments. By implementing an algorithm that can identify toxic comments automatically, we can significantly enhance productivity, enabling agents to focus on more critical cases instead of reviewing all comments. 

This solution will not only improve operational efficiency but also ensure faster responses to harmful content, fostering a safer and higher-quality online environment. It offers a scalable approach to content moderation, providing a valuable tool for businesses that need to manage large datasets and optimize their processes.

## Table of Contents

1. [Introduction](#Introduction)
2. [Preprocessing and EDA](#Preprocessing_and_EDA)
3. [Modeling_phase](#Modeling_phase)
4. [Interpretability](#Interpretability)
5. [Next_steps](#Next_Steps)
6. [License](#License)

## Introduction

The dataset for this project contains various types of comments, with a significant imbalance between toxic and non-toxic comments. Toxic comments are relatively rare on social media platforms, which presents a class imbalance challenge for the model.

The comments in the dataset have been labeled by human raters to identify toxic behavior. These toxic categories include:
- **Toxic**  
- **Severe Toxic**  
- **Obscene**  
- **Threat**  
- **Insult**  
- **Identity Hate**  

## Preprocessing_and_EDA

In the preprocessing stage, we analyze the dataset from various perspectives. Key observations include:

- The dataset is highly imbalanced, requiring the use of techniques such as under-sampling, over-sampling, or class weights to balance the data.
- There were no discernible patterns in terms of character count, word length, or other simple metrics that could suggest predictability in the comments.
- We analyzed approximately 28,000 tokens containing a variety of words.
- Several challenging words were identified, particularly those related to the context of toxicity. However, some anomalous tokens were found, such as the word "Wikipedia."
- Unusual characters from other languages were also noted, though many had already been translated, with the majority of the comments being in English.
- Using PCA and t-SNE, we identified clusters and relationships between words, despite the high dimensionality of the data.

In conclusion, these are the steps we used to filter the dataset to reduce noise:  
- Removed HTML content from the rows.  
- Converted emojis into words.  
- Filtered out comments exceeding 256 tokens to prepare the data for training purposes.

The **dataset is split** as follows: we have two files, one for training and another for testing. The training file is further split into training and validation sets. As for the test set, it is left unchanged, except for removing entries without any category predictions.
- Train set = 28k samples (114k originally)
- validation set = 28k samples 
- test set = 63k samples 

For further analysis, refer to the full [EDA Analysis](./EDA_analysis.md) and the linked notebooks:

- [001_EDA](./001_EDA.ipynb)
- [002_EDA_Advance](./002_EDA_Advance.ipynb)

## Modeling_phase

The initial model performs well on the majority class (non-toxic comments), but improvements are 
necessary for rare categories such as **severe_toxic**, **threat**, and **identity_hate**. We plan to experiment with different thresholds for these categories and generate synthetic data to improve performance on the imbalanced classes.

While the model achieves an overall decent accuracy, the strong performance on non-toxic comments 
contributes significantly to this result. The recall rate of 68% is a good starting
point, but there is room for improvement, particularly in detecting rare toxic categories.

We select the **roBerta** as the final model for performance also, for that we decide to use **AdamW** for the optimizer and   
**BCEWithLogitsLoss** with the weights for balancing the dataset.

Here are some parameters that we can see from the model:
```python
Best hyperparameters: 
{
  'batch_size': 16,
  'lr': 2.057626963810742e-06,
  'warmup': 0.2410014240231825,
  'w_decay': 8.747541196471078e-09,
  'dropout_vals': 0.3789390401864389
}
```

For detailed modeling experiments, refer to the [Model Experiments](./model_experiments.md) and the linked notebooks:

- [003_model_experiment.ipynb](003_model_experiment.ipynb)
- [004_model_experiment_undersampling.ipynb](004_model_experiment_undersampling.ipynb)
- [005_model_experiment_use_weights_sample_set.ipynb](005_model_experiment_use_weights_sample_set.ipynb)
- [006_model_experiment_use_weights.ipynb](006_model_experiment_use_weights.ipynb)
- [007_model_experiment_threshold.ipynb](007_model_experiment_threshold.ipynb)
- [008_model_experiment_oversampling.ipynb](008_model_experiment_oversampling.ipynb)
- [010_model_experiment_roberta_longer_epochs.ipynb](010_model_experiment_roberta_longer_epochs.ipynb)
- [011_model_experiment_roberta_results.ipynb](011_model_experiment_roberta_results.ipynb)
- [012_model_experiment_roberta.ipynb](012_model_experiment_roberta.ipynb)
- [013_model_experiment_roberta_balance_non_toxic.ipynb](013_model_experiment_roberta_balance_non_toxic.ipynb)
- [014_model_experiment_roberta_hyperparameters.ipynb](014_model_experiment_roberta_hyperparameters.ipynb)
- [015_model_experiment_roberta_more_epochs.ipynb](015_model_experiment_roberta_more_epochs.ipynb)
- [016_syntetic_data.ipynb](016_syntetic_data.ipynb)
- [017_model_experiments_synthetic_data.ipynb](017_model_experiments_synthetic_data.ipynb)
- [018_model_Threshold_setup.ipynb](018_model_Threshold_setup.ipynb)
- [019_model_manually_move_threshold.ipynb](019_model_manually_move_threshold.ipynb)

Given the imbalanced proportions, we primarily used class weights to handle the imbalance, as shown in the following imbalance chart:

![imbalance_proportions.png](img/imbalance_proportions.png)

Here it's a table for some experiment and their results so we can see how we handle them.

| Notebook                                 | Model Name     | Base Model | Loss Function      | Optimization | Balancing Technique                                   | Epochs | Train Recall | Train Accuracy | Train Loss | Validation Recall | Validation Accuracy | Validation Loss | Test Recall                                        | Test Accuracy                                         | ROC  | Recall Global | Accuracy Global |
|------------------------------------------|----------------|------------|--------------------|--------------|------------------------------------------------------|--------|--------------|----------------|------------|-------------------|---------------------|----------------|--------------------------------------------------|--------------------------------------------------|-------|---------------|----------------|
| 003_model_experiment                     |                | BertModel  | BCEWithLogitsLoss  | Adam         | Nothing                                              | 200    |              | 100.00%        | 0.0703101034   |                   | 89.81%             |                |                                                  |                                                  |       |               |                |
| 004_model_experiment_undersampling       | 002_model_bert | BertModel  | BCEWithLogitsLoss  | AdamW        | Weights                                              | 200    |              | 60.71%         | 0.0008103143677|                   | 0.31%              | 0.4963846411   |                                                  |                                                  |       |               |                |
| 005_model_experiment_use_weights_sample_set | 005_model_bert | BertModel  | BCEWithLogitsLoss  | AdamW        | Weights                                              | 5      | 5.75%        | 96.64%         | 0.3821     | 8.59%             | 90.06%             | 0.3691         |                                                  |                                                  |       |               |                |
| 006_model_experiment_use_weights         | 006_model_bert | BertModel  | BCEWithLogitsLoss  | AdamW        | Weights                                              | 6      | 0.2410178716 | 97.85%         | 0.1967672146| 0.2474            | 0.9117             | 0.1835         |                                                  |                                                  |       |               |                |
| 007_model_experiment_threshold           | 007_model_bert_threshold | BertModel | BCEWithLogitsLoss | AdamW | Weights                                              | 4      | 0.041278     | 0.968326       | 0.280624   | 0.085774          | 0.969149           | 0.264581       |                                                  |                                                  |       |       `        |                |
| 008_model_experiment_oversampling        | 002_Bert       | BertModel  | BCEWithLogitsLoss  | AdamW        | Weights * 2 in undersampling, Translation language  | 1      | 0.2355       | 0.166          | 0.6101     | 1                 | 0.0381             | 0.6101         |                                                  |                                                  |       |               |                |
| 010_model_experiment_roberta             | Roberta_010    | Roberta    | BCEWithLogitsLoss  | AdamW        | Nothing                                              | 27     | 0.979167     | 0.979167       | 0.047904   | 0.933333          | 0.933333           |                |                                                  |                                                  |       |               |                |
| 011_model_experiment_roberta             |                | Roberta    | BCEWithLogitsLoss  | AdamW        | Nothing                                              | 1      | 0.750501     | 0.416667       | 0.416667   | 0.472222          | 0.472222           | 0.760931       | Treat: 63%; Identity Hate: 47%; Severe Toxic: 27%  | Treat: 52%; Identity Hate: 44%; Severe Toxic: 28%   |       |               |                |
| 012_model_experiment_roberta             | model_002      | Roberta    | BCEWithLogitsLoss  | AdamW        | Nothing                                              | 4      | 0.833333     | 0.833333       | 0.341658   | 1                 | 1                  | 0.015301       | Treat: 93%; Identity Hate: 91%; Severe Toxic: 98%  | Severe Toxic: 95%; Treat: 92%; Identity Hate: 86%   |       |               |                |
| 013_model_experiment_roberta_balance_non_toxic | model_013  | Roberta    | BCEWithLogitsLoss  | AdamW        | Weights                                              | 5      | 100.00%      | 100.00%        | 0.004656   | 0.979167          | 0.979167           | 0.019967       | Severe Toxic: 94%; Treat: 19%; Identity Hate: 77%  | Severe Toxic: 93%; Treat: 22%; Identity Hate: 65%   |       |               |                |
| 014_model_experiment_roberta_treshold    | model_014      | Roberta    | BCEWithLogitsLoss  | AdamW        | Weights                                              | 1      | 0.775        | 0.775          | 0.11       | 0.962             | 0.962              | 0.0643         | Severe Toxic: 56%; Treat: 56%; Identity Hate: 57%  | Severe Toxic: 48%; Treat: 63%; Identity Hate: 60%   |       |               |                |
| 015_model_experiment_roberta_treshold    | model_015      | Roberta    | BCEWithLogitsLoss  | AdamW        | Weights                                              | 7      | 1            | 1              | 0.004728   | 0.989583          | 0.989583           | 0.015964       | Severe Toxic: 98%; Treat: 70%; Identity Hate: 94%  | Severe Toxic: 97%; Treat: 58%; Identity Hate: 90%   | 0.68  | 0.97          |                |
| 017_model_experiments_synthetic_data     | model_017      | Roberta    | BCEWithLogitsLoss  | AdamW        | Weights                                              | 4      | 0.871795     | 0.871795       | 0.07063    | 0.979167          | 0.979167           | 0.021978       | Severe Toxic: 95%; Treat: 78%; Identity Hate: 84%  | Severe Toxic: 82%; Treat: 64%; Identity Hate: 65%   | 0.53  | 0.96          |                |
| 018_model_Threshold_setup                | model 018      | Roberta    | BCEWithLogitsLoss  | AdamW        | Nothing                                              |        |              |                |            |                   |                     |                | Severe Toxic: 98%; Treat: 70%; Identity Hate: 94%  | Severe Toxic: 97%; Treat: 62%; Identity Hate: 91%   |       |               |                |
| 018_model_Threshold_setup                | model 018      | Roberta    | BCEWithLogitsLoss  | AdamW        | Nothing                                              |        |              |                |            |                   |                     |                | Severe Toxic: 95%; Treat: 78%; Identity Hate: 84%  | Severe Toxic: 88%; Treat: 73%; Identity Hate: 74%   |       |               |                |
| 019_model_manually_move_threshold        | model 019      | Roberta    | BCEWithLogitsLoss  | AdamW        | Nothing                                              |        |              |                |            |                   |                     |                |                                                  |                                                  | 0.76  | 0.94          |                |


The final model used **RoBERTa**, with the following classification report:

Finally, we used these thresholds to achieve better predictability.
```
class_thresholds = {
    "toxic": 0.50,
    "severe_toxic": 0.10,
    "obscene": 0.50,
    "threat": 0.50,
    "insult": 0.50,
    "identity_hate": 0.10
}
```

By adjusting the thresholds, we improved the performance by 6%, which enhanced the model's predictability.

```
Global Classification Report:
              precision    recall  f1-score   support

           0       0.99      0.95      0.97    369370
           1       0.36      0.76      0.49     14498

    accuracy                           0.94    383868
   macro avg       0.68      0.85      0.73    383868
weighted avg       0.97      0.94      0.95    383868
```
This was test it in the verify dataset.

### Model Insights

The model achieves a high overall accuracy of 94%. It performs exceptionally well in classifying 
non-toxic comments (Class 0) with high precision, recall, and F1 score. However, 
its performance on detecting toxic comments (Class 1) is less accurate, with a precision of 0.36 and
recall of 0.76. While the weighted average metrics are strong, the macro average indicates that 
the model needs further refinement, especially in identifying toxic comments.

## Interpretability

To understand how the model makes its predictions, we review two examples:

### Example 1:
**Comment:** "How dare you vandalize that page about the HMS Beagle! Don't vandalize again, demon!"  
**Predicted Labels:** ['toxic']  
**Probabilities:** [0.789, 0.032, 0.042, 0.095, 0.140, 0.056]  

This example appears to be correctly classified as 'toxic'.

### Example 2:
**Comment:** "DJ Robinson is gay as hell! he sucks his dick so much!!!!!"  
**Predicted Labels:** ['toxic', 'obscene', 'insult']  
**Probabilities:** [0.982, 0.193, 0.932, 0.127, 0.707, 0.181]  

In this case, the model correctly predicts multiple categories of toxicity.

## Next_Steps

Next, we plan to experiment with different thresholds for the categories, both globally and individually. We will also generate synthetic data to improve the detection of rare toxic comments. Additionally, gathering more data for imbalanced classes could help further improve model performance.

## Citation

- Project: **Toxic Comment Classification Challenge**
- Publisher: Kaggle
- Year: 2018
- URL: [Kaggle Toxic Comment Challenge](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge)

## License

Please refer to the [License.md](./License) file for licensing details.

---

This revision improves readability and flow while retaining all key details. Let me know if you need any further adjustments!
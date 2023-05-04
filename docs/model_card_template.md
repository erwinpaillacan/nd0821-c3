# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
RandomForestClassifier trained with scikit-learn library and Random State 42
## Intended Use
to predict salary classes based on demographic and job-related features from U.S. Census data.
## Training Data
sourced from the U.S. Census data with various features including workclass, education, marital status, occupation, relationship, race, sex, and native country.
## Evaluation Data
also sourced from the U.S. Census data with an 80-20 train-test split performed on the dataset.
## Metrics
weighted precision, recall, and F1 score.
precision: 0.7974452554744526
recall: 0.5500314663310258
fbeta: 0.651024208566108
## Ethical Considerations
potential biases in the model's predictions due to sensitive demographic information such as race, sex, and native country, which could perpetuate existing inequalities and discrimination in income distribution and job opportunities.
## Caveats and Recommendations
class imbalance issue, limited generalizability, recommended fine-tuning with more representative data, periodic retraining, caution in decision-making reliance, and combining the model's output with other sources of information and expert opinions.

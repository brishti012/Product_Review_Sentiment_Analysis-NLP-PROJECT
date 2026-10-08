Amazon Product Review Sentiment Analysis
An NLP project for classifying Amazon product reviews into Negative,
Neutral, and Positive sentiment using classical machine learning and
deep learning, with model evaluation, subgroup auditing, error analysis,
and explainability.

Team Members
- Brishti Das --- 25030422035
- Pushyami Dommaraju --- 25030422045
- Mahee Aggrawal --- 25030422064

Project Overview
Customer reviews contain valuable information about user experience, but
manually analysing thousands of reviews is time-consuming. This project
applies Natural Language Processing (NLP) to automatically classify
Amazon product reviews into three sentiment categories:
  Rating       Sentiment
  1--2 stars   Negative
  3 stars      Neutral
  4--5 stars   Positive
The project does not evaluate models only on overall accuracy. Because
the dataset is highly imbalanced, the analysis places particular
emphasis on Macro F1, which gives equal importance to all three
sentiment classes.
The project also investigates how model performance changes with review
length, examines common prediction errors, and uses model-specific
techniques to understand predictions.

Objectives
1. Clean and preprocess Amazon product review text.
2. Convert star ratings into three sentiment classes.
3. Build a classical NLP baseline using TF-IDF + Logistic
   Regression.
4. Build a deep-learning sentiment classifier using BiLSTM.
5. Evaluate models using Accuracy, Precision, Recall, Macro F1, and
   Weighted F1.
6. Audit model performance across Short, Medium, and Long reviews.
7. Perform confusion-matrix and error analysis.
8. Examine model predictions and confidence.
9. Interpret Logistic Regression predictions using TF-IDF feature
   coefficients.
10. Identify limitations and areas for future improvement.

Dataset
Dataset: Datafiniti Amazon Consumer Reviews
Initial dataset size:
- 34,660 reviews
- 21 columns
Important fields include:
- reviews.text
- reviews.title
- reviews.rating
- reviews.date
- brand
- categories
- manufacturer
Sentiment Distribution
  Sentiment     Reviews   Percentage
  Negative          812        2.34%
  Neutral         1,499        4.33%
  Positive       32,316       93.33%
The dataset is therefore severely imbalanced, with Positive reviews
representing approximately 93% of the observations.
Data Quality
The preprocessing analysis identified:
- 1 missing review text
- 33 missing ratings
- 0 complete duplicate rows
- 1 duplicate review text
- Several metadata fields with substantial missingness
Only fields required for the sentiment task were used for essential
missing-value handling.

Data Preprocessing
The preprocessing pipeline was:
Amazon Reviews
      ↓
Combine Review Title + Review Text
      ↓
Handle missing required fields
      ↓
Lowercase
      ↓
Remove URLs
      ↓
Remove HTML tags
      ↓
Normalize whitespace
      ↓
Keep stopwords
      ↓
Remove duplicate cleaned review text
      ↓
Create review-length groups
      ↓
Stratified Train / Validation / Test Split
Important preprocessing decisions
Combining title and review text
The review title and review body were combined to provide the models
with more textual information.
Lowercasing
Text was converted to lowercase to reduce unnecessary variation between
uppercase and lowercase versions of the same word.
URLs and HTML
URLs and HTML tags were removed because they generally do not contribute
useful sentiment information for this task.
Stopwords were retained
Stopwords were intentionally not removed because negations such as
"not" can change sentiment.
For example:
good
and
not good
have very different sentiment.
Duplicate-text leakage prevention
Duplicate cleaned review text was removed before the final data split to
prevent identical reviews from appearing across training, validation,
and test sets.
Final overlap checks showed:
Train ∩ Validation = 0
Train ∩ Test       = 0
Validation ∩ Test  = 0
Review Length Groups
For subgroup auditing:
- Short: ≤20 words
- Medium: 21--50 words
- Long: >50 words
Data Split
A stratified split was used:
- 70% Training
- 15% Validation
- 15% Testing
Stratification helps preserve the sentiment-class proportions across the
three datasets.

Feature Representation
TF-IDF
For the classical model, reviews were converted into numerical features
using TF-IDF (Term Frequency--Inverse Document Frequency).
The implementation used:
- Maximum features: 20,000
- Unigrams and bigrams
- min_df = 2
- max_df = 0.95
- Sublinear TF scaling
Using unigrams and bigrams allows the model to learn both individual
words and short phrases such as:
great
terrible
not good
not worth
love it

Models
1. TF-IDF + Logistic Regression
This was used as the classical machine-learning baseline.
Pipeline:
Review Text
     ↓
TF-IDF
     ↓
Unigrams + Bigrams
     ↓
Logistic Regression
     ↓
Negative / Neutral / Positive
The model used:
class_weight="balanced"
to reduce the effect of the severe class imbalance.
An important advantage of Logistic Regression in this project is its
interpretability: its learned TF-IDF coefficients can be inspected to
identify features associated with each sentiment class.
2. BiLSTM
A Bidirectional Long Short-Term Memory (BiLSTM) network was
implemented as the deep-learning model.
Pipeline:
Review Text
     ↓
Tokenization
     ↓
Integer Sequences
     ↓
Padding
     ↓
Embedding
     ↓
Bidirectional LSTM
     ↓
Dense Layer
     ↓
Softmax
     ↓
Negative / Neutral / Positive
Key settings included:
- Maximum sequence length: 100
- Embedding dimension: 128
- LSTM units: 64
- Dense layer
- Dropout
- Adam optimizer
- Early stopping

Model Evaluation
Because the dataset is highly imbalanced, the project considers:
- Accuracy
- Precision
- Recall
- F1-score
- Macro F1
- Weighted F1
- Confusion Matrix
Why Macro F1?
Accuracy can be misleading when one class dominates the dataset.
Since approximately 93% of reviews are Positive, a model can obtain high
accuracy while performing poorly on Negative and Neutral reviews.
Macro F1 gives each sentiment class equal importance, making it a
more informative measure of balanced performance.
Final Model Results
  Model                                   Accuracy          Macro F1       Weighted F1
  TF-IDF + Logistic Regression              89.87%        61.89%            91.45%
  BiLSTM                                    93.86%            47.49%            92.36%
  DistilBERT                       Not implemented   Not implemented   Not implemented
Key Finding
BiLSTM achieved higher overall Accuracy and Weighted F1, but
Logistic Regression achieved substantially higher Macro F1.
This shows why a single metric should not be used to judge performance
on a severely imbalanced dataset.
For this project, Macro F1 is particularly important because it reflects
how well the model handles all three sentiment classes rather than
primarily the Positive class.

Confusion Matrix & Error Analysis
The models were evaluated using confusion matrices to identify which
sentiment classes were most frequently confused.
Logistic Regression
The dominant confusion pattern was:
Positive → Neutral
followed by:
Neutral → Positive
BiLSTM
The dominant confusion pattern was:
Neutral → Positive
followed by:
Negative → Positive
These different error patterns show that the two models do not fail in
exactly the same way.
Error analysis was also performed by review length and through
prediction confidence.

Review-Length Audit
Macro F1 was calculated separately for Short, Medium, and Long reviews.
  Review Length     Logistic Regression   BiLSTM
  Short                          0.6338   0.4321
  Medium                         0.6091   0.4556
  Long                           0.6014   0.5020
Observation
Logistic Regression performs better across all three review-length
groups.
Its Macro F1 decreases slightly as review length increases, while BiLSTM
shows a slight improvement on longer reviews.
These results describe observed performance differences; they do not
establish that review length itself causes the performance change.

Explainability
Logistic Regression --- TF-IDF Coefficients
The Logistic Regression model was interpreted using its learned TF-IDF
coefficients.
Examples of strongly associated features included:
Negative: - not - returned - terrible - not worth - never
Neutral: - ok - okay - but - decent - however
Positive: - great - love - awesome - excellent - perfect
These coefficients provide an interpretable view of which textual
features the model associates with different sentiment classes.
BiLSTM --- Prediction Confidence
For the BiLSTM, prediction-level analysis was used to inspect confidence
and incorrect predictions.
The analysis showed that confidence can help identify uncertain cases,
but:
Confidence does not guarantee correctness.

Limitations
1. Severe class imbalance: Positive reviews dominate the dataset.
2. Rating-based labels: Star ratings are used as a proxy for
   textual sentiment and may not perfectly represent the sentiment
   expressed in the review.
3. Mixed 3-star reviews: Neutral reviews can contain both positive
   and negative opinions.
4. Sarcasm and context: Models can struggle with sarcasm,
   ambiguity, and context-dependent language.
5. Dataset representation: The dataset may not represent every
   customer, product, writing style, or review platform.
6. Computational constraints: DistilBERT was considered but not
   implemented because practical fine-tuning was not feasible within
   the project resources.

Future Scope
Possible extensions include:
- Fine-tuning DistilBERT or another transformer model with sufficient
  computational resources.
- Exploring contextual embeddings and transformer-based architectures.
- Improving minority-class performance through additional
  imbalance-handling strategies.
- Expanding the dataset to include more diverse products and review
  styles.
- Improving handling of mixed sentiment and context-dependent
  language.
- Performing more detailed explainability analysis using methods such
  as SHAP or LIME.
- Investigating additional subgroup characteristics beyond review
  length.

Technologies Used
- Python
- Pandas
- NumPy
- Scikit-learn
- TensorFlow / Keras
- Matplotlib
- Seaborn
- Jupyter Notebook / Google Colab

Project Structure
Amazon-Review-Sentiment-Analysis/
│
├── NLP_project_2.ipynb
├── README.md
│
├── data/
│   └── amazon dataset nlp.csv
│
├── presentation/
│   └── Amazon_Sentiment_Pitch.pptx
│
└── results/
    ├── confusion_matrix/
    ├── model_comparison/
    ├── explainability/
    └── error_analysis/
Update the folder/file names above to match the actual structure of
the GitHub repository.

Project Structure: Build → Audit → Pitch
This project follows a Build--Audit--Pitch framework.
Build
- Dataset preparation
- Text preprocessing
- Stratified splitting
- TF-IDF + Logistic Regression
- BiLSTM
- Model evaluation
Audit
- Class-wise performance
- Review-length subgroup analysis
- Confusion matrix
- Error analysis
- Prediction confidence
- Logistic Regression feature explainability
- Limitations
Pitch
The main project finding is:
Overall accuracy alone can be misleading in a highly imbalanced
sentiment dataset. Logistic Regression achieved better balanced
performance according to Macro F1, while BiLSTM achieved higher
overall accuracy and Weighted F1.

Team
Brishti Das
25030422035
Pushyami Dommaraju
25030422045
Mahee Aggrawal
25030422064

References
The project uses the following types of resources:
- Datafiniti Amazon Consumer Reviews dataset
- Scikit-learn documentation
- TensorFlow / Keras documentation
- NLP and sentiment-analysis literature
- Documentation for Python libraries used in the implementation
Specific dataset, paper, and documentation links should be added to this
section based on the sources actually used by the team.

Academic Project
This repository contains the implementation and analysis for an academic
NLP project on Amazon Product Review Sentiment Analysis.

# Sentiment-Analysis-on-Rotten-Tomatoes-Reviews
This repository contains code and documentation for the Sentiment Analysis project on Rotten Tomatoes reviews by Beatriz Correia (Myself) and Luís Pereira. It classifies review sentiment using techniques like lexicon-based methods, machine learning classifiers, and transformer-based models.

## Introduction
Sentiment analysis is a crucial task in Natural Language Processing (NLP) that involves extracting subjective information from text. This project focuses on classifying the sentiment of movie reviews from Rotten Tomatoes into positive, negative, or neutral categories. The dataset used consists of reviews in English, divided into training and testing sets. An exploratory data analysis was followed by various preprocessing techniques. Classification models were developed and evaluated based on precision, recall, and F1-score. Notable models include DistilBERT and OpenAI's ChatGPT for generative sentiment classification. Results are discussed in detail, highlighting the advantages and limitations of each approach and potential future research directions.

## Business Understanding
The goal of this project is to understand and classify user opinions on movies by analyzing their reviews. Accurate sentiment analysis can provide valuable insights for movie producers, marketers, and review platforms, enabling them to gauge public opinion and make data-driven decisions.
Key objectives include:
- Classifying movie reviews as positive, negative, or neutral.
- Comparing different sentiment analysis techniques.
- Evaluating the performance of various models.

## Data Understanding
- The project utilizes film reviews in English from Rotten Tomatoes, divided into training (6800 reviews) and test (1729 reviews) datasets in TSV format. Each dataset contains three attributes: review ID, assigned sentiment (positive, negative, neutral), and review text. A 'sentiment_numeric' column was created with values 1, -1, and 0 corresponding to positive, negative, and neutral sentiments. The test set includes 714 positive, 671 negative, and 344 neutral reviews, while the training set contains 2888 positive, 2601 negative, and 1311 neutral reviews. No missing values are present.
  
Each dataset contains the following columns:
|Column | Type| Description |
|:---:|:---:| :---:| 
| **Id** |String| Review Id|
| **Sentiment** |String| Sentiment label (positive, negative, neutral) |
| **Review** | String | The text of the review |

Note: The dataset could not be shared due confidentiality. 

## Data Preparation
- Imported the dataset using Pandas. Added headers to the datasets as they initially lacked them. Checked for and confirmed the absence of missing values.

## Exploratory Data Analysis (EDA)
Descriptive statistics were computed for the sentiment distribution.
Visualizations were created to understand the data distribution.

## Data Preprocessing
Data preprocessing involved multiple steps, including normalization (lowercasing and stemming), POS tagging, tokenization, named entity recognition, stopword removal, lemmatization, and negation handling. Three popular libraries were integrated for different preprocessing techniques. The data was processed in stages, with new columns created for each applied technique, using Google Colab to handle computational intensity.

## Modeling
### Baseline Models
- VaderSentiment: Used for initial sentiment classification, achieved 52% precision, 51% recall, and 51% F1-score.
- Stanza: Applied CNN-based sentiment classifier, achieved 83% precision, 82% recall, and 82% F1-score.

### Lexicon-Based Classifier
A sentiment classifier was developed using the 'NRC Word-Emotion Association Lexicon,' converted to a dictionary format for performance optimization. Reviews were classified based on cumulative sentiment scores, with values mapped to sentiment labels. Various preprocessing techniques were applied, including case folding, contraction expansion, tokenization, POS tagging, stopword removal, lemmatization, and negation handling. Performance was evaluated using precision, recall, and F1-score.

### Machine Learning Classifiers
#### Transformations:
- Bag of Words: Transformed text into a matrix of token counts.
- TF-IDF: Measured the importance of words in the reviews.
- Word Embeddings: Represented words as vectors in a multidimensional space.
  
#### Classifiers including:
- Logistic Regression: Modeled log-odds of sentiments.
- Multinomial Naive Bayes: Calculated probabilities of sentiments.
- Support Vector Machine (SVM): Found optimal hyperplane for classification.
  
### Transformer-Based Models
The pre-trained "DistilBERT" model was applied with a sentiment-analysis pipeline. Initial testing showed precision of 75%, recall of 38%, and F1-score of 49%. Fine-tuning the model on the dataset improved performance, converting sentiment labels and tokenizing reviews. Training was conducted using the Hugging Face API, optimizing training arguments for better results.

### Generative Models
The final step used OpenAI's ChatGPT for generative sentiment classification. The model was applied to a dataset of 421 Portuguese laptop reviews, with predictions labeled as "pos" or "neg." The model utilized GPT-3.5 Turbo, adjusted for higher performance and efficiency. Results were documented, and outputs were evaluated against a sample from the training dataset.
Note: The dataset used for the generative models could not be added due to confidentiality purposes.

### Evaluation
Because the reviews can be classified in 3 different ways (positive, negative and neutral) the models were evaluated by using the following metrics:
- Precision: The accuracy of positive predictions.
- Recall: The ability to capture all positive instances.
- F1-Score: The harmonic mean of precision and recall.

### Best Performing Models
- Stanza Baseline: Achieved the highest performance with Precision: 0.83, Recall: 0.82, and F1-Score: 0.82.
- DistilBERT Fine-Tuned: Showed balanced performance with Precision: 0.66, Recall: 0.66, and F1-Score: 0.66.
- Machine Learning Classifiers: Logistic Regression and Naive Bayes with TF-IDF demonstrated competitive results.

### Conclusion
The project successfully explored multiple approaches to sentiment analysis on Rotten Tomatoes reviews. Key takeaways include:
- Lexicon-based methods are less effective compared to machine learning classifiers.
- Transformer-based models, especially with fine-tuning, offer robust performance.
- Handling neutral sentiment remains challenging due to class imbalance.
- Future work could involve experimenting with deep learning models, such as convolutional neural networks (CNNs), and further refining preprocessing techniques like negation handling.

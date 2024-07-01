# Sentiment-Analysis-on-Rotten-Tomatoes-Reviews
This repository contains code and documentation for the Sentiment Analysis project on Rotten Tomatoes reviews by Beatriz Correia (Myself) and Luís Pereira. It classifies review sentiment using techniques like lexicon-based methods, machine learning classifiers, and transformer-based models.

## Introduction
Sentiment analysis is a crucial task in Natural Language Processing (NLP) that involves extracting subjective information from text. This project focuses on classifying the sentiment of movie reviews from Rotten Tomatoes into positive, negative, or neutral categories. The dataset used consists of reviews in English, divided into training and testing sets.

## Business Understanding
The goal of this project is to understand and classify user opinions on movies by analyzing their reviews. Accurate sentiment analysis can provide valuable insights for movie producers, marketers, and review platforms, enabling them to gauge public opinion and make data-driven decisions.
Key objectives include:
- Classifying movie reviews as positive, negative, or neutral.
- Comparing different sentiment analysis techniques.
- Evaluating the performance of various models.

## Data Understanding
The dataset could not be shared due confidentiality. It consists of:
- rotten_tomatoes_train.tsv: Training set with 6800 reviews.
- rotten_tomatoes_test.tsv: Testing set with 1729 reviews.
Each dataset contains the following columns:

id: Review ID.
sentiment: Sentiment label (positive, negative, neutral).
review: The text of the review.
To facilitate analysis, an additional column sentiment_numeric was created, mapping the sentiment labels to numeric values (positive: 1, negative: -1, neutral: 0).

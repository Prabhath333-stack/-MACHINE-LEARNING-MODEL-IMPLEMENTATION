# -MACHINE-LEARNING-MODEL-IMPLEMENTATION

"COMPANY NAME": CODETECH IT SOLUTIONS

"NMAE" : PRABHATH THUMMALA

"INTERN ID" : CT04DG393

"DOMAIN" : PYTHON

"DURATION" : 4 WEEKS

"MENTOR" : NEELA SANTHOSH

DESCRIPTION:-
This project demonstrates how to build a Spam Email Detection Model using the Scikit-learn library in Python within Google Colab. The goal is to classify messages as either spam or ham (not spam) using machine learning.

The process begins by importing essential libraries such as pandas and NumPy for data manipulation, matplotlib and seaborn for visualization, and several modules from Scikit-learn for model building and evaluation.

Next, the dataset is uploaded using files.upload() from google.colab. The dataset used here is the SMS Spam Collection Dataset, which contains labeled SMS messages. After uploading, the file is read using pandas.read_csv() and unnecessary columns are dropped, retaining only the message text and label. The labels ‘ham’ and ‘spam’ are then converted into binary format (0 and 1), which is required for machine learning classification tasks.

For text data preprocessing, the CountVectorizer from sklearn.feature_extraction.text is used. This tool converts the text messages into a bag-of-words representation—a format that transforms text into numerical vectors based on word frequency, allowing the model to process the data effectively.

The dataset is then split into training and testing sets using train_test_split. This ensures that the model is trained on a portion of the data and evaluated on unseen data to measure its real-world performance.

A Naive Bayes classifier (MultinomialNB) is chosen due to its simplicity and efficiency with text data. The model is trained using model.fit() and predictions are made on the test set using model.predict().

The model's performance is evaluated using several metrics:

Accuracy score gives the overall correctness.

Classification report provides precision, recall, and F1-score for both spam and ham classes.

Confusion matrix is visualized using seaborn's heatmap, showing correct and incorrect predictions.

Finally, the model is tested on a new custom message to demonstrate real-time prediction, which outputs whether the message is spam or not.

Overall, this code provides a complete pipeline for spam detection, from data upload and preprocessing to model training and evaluation, all done interactively and efficiently in Google Colab.

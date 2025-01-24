import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score
import scikitplot

#Load the training and testing data.
file_train=pd.read_csv('train_file.csv')
file_test=pd.read_csv('test_file.csv')

#Extract the text(X) and label(Y) for both training and testing.
X_train_text=file_train['text']
X_test_text=file_test['text']
Y_train=file_train['label']
Y_test=file_test['label']

#CountVectorizer function. To convert a collection of text documents to a matrix of token counts.
vectorizer_v1 = CountVectorizer(min_df=10,stop_words='english')
vectorizer_v1.fit(X_train_text)#Create a dictionary of all the vocabulary tokens present in the raw documents.
X_train_v1 = vectorizer_v1.transform(X_train_text)#Transform documents to document-term matrix.

#Using 'Support Vector Machines (SVMs)' Classifier and then fit the classifier on the vectorized training data.
model_v1=SVC() 
model_v1.fit(X_train_v1, Y_train)

#Create a prediction pipeline
prediction_pipeline_v1 = make_pipeline(vectorizer_v1, model_v1)

#Apply the pipeline to predict the labels of the testing data.
predictions_v1 = prediction_pipeline_v1.predict(X_test_text)
#Print the accuracy and the percentage of the model by using 'accuracy_score' function.
print('The accuracy of this model is:',accuracy_score(Y_test, predictions_v1))
print('The accuracy percentage is: {:.2%}'.format(accuracy_score(Y_test, predictions_v1)))
#Create a Confusion Matrix. 
class_names=['label_0','label_1']
scikitplot.metrics.plot_confusion_matrix([class_names[i] for i in Y_test], 
                                    [class_names[i] for i in predictions_v1],  
                                    title="Confusion Matrix", 
                                    cmap="Greens",  
                                    figsize=(4,4))
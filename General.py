import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import neighbors
from sklearn.metrics import confusion_matrix, f1_score
from scipy.spatial.distance import euclidean, cityblock

# a DataFrame which is going to be used to hold all results we get is initialised
results = pd.DataFrame(columns=['Split Variation', 'k', 'Distance Metric', 'Confusion Matrix', 'F1 score'])

# data is read from the csv file
data = pd.read_csv('./csvdata.csv')

# the data we want to split into training and test sets is put into variables for better readability
features = data[['Formant 1', 'Formant 2', 'Formant 3']].values
labels = data[['Vowel Phoneme']].values.ravel()

# initialize a dictionary to store the confusion matrices for each split variation and distance metric
confusion_matrices = {}

# clustering is run for 2 different distance metrics, to identify the best approach
for metric in [euclidean, cityblock]:
    # for every distance metric, the algorithm is run 5 times, with 5 different test and training sets.
    for x in range(5):
        # removed state = 0 from the code provided to make different variations for each iteration of the loop
        X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.25, stratify=data[['Gender (M/F)']])
        
        # k is run for k in the range of 3 to 7, so that the optimal value of k is selected
        for k in range(3, 8):
            # initialising the classifier object
            classifier = neighbors.KNeighborsClassifier(n_neighbors=k, metric=metric)
            # training the classifier on our training data
            classifier.fit(X_train, y_train)
            # predicting values for our test set
            y_pred = classifier.predict(X_test)
            # confusion matrices and F1 scores are computed
            conf_matrix = confusion_matrix(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted')
            # results obtained appended to the dataframe
            results = results.append({'Split Variation': x+1, 'k': k, 'Distance Metric': metric.__name__, 'Confusion Matrix': conf_matrix, 'F1 score': f1}, ignore_index=True)
            
            # store the confusion matrix for further analysis
            confusion_matrices[(metric.__name__, x+1, k)] = conf_matrix

# the results are finally exported to an excel sheet, which is easier to visualize in the report
results.to_excel('./results.xlsx')

# analyze the confusion matrices to identify the vowel-based phonemes that produce the most confusion
most_confused_phonemes = {}

for key, matrix in confusion_matrices.items():
    metric_name, split_variation, k = key
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            true_label = labels[i]
            predicted_label = labels[j]
            if true_label != predicted_label:
                if true_label not in most_confused_phonemes:
                    most_confused_phonemes[true_label] = {}
                if predicted_label not in most_confused_phonemes[true_label]:
                    most_confused_phonemes[true_label][predicted_label] = 0
                most_confused_phonemes[true_label][predicted_label] += matrix[i, j]

print(most_confused_phonemes)

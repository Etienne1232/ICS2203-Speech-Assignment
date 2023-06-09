import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import neighbors
from sklearn.metrics import confusion_matrix, f1_score
from scipy.spatial.distance import euclidean, cityblock

# a DataFrame which is going to be used to hold all results we get is initialised
results = pd.DataFrame(columns=['Gender', 'Split Variation', 'k', 'Data Type', 'Distance Metric', 'Confusion Matrix', 'F1 score'])

# data is read from the csv file
data = pd.read_csv('./csvdata.csv')
male_data = data.loc[data['Gender (M/F)'] == 'M']
female_data= data.loc[data['Gender (M/F)'] == 'F']

for gender in [male_data, female_data]:
    # the data we want to split into training and test sets is put into variables for better readability
    features = gender[['Formant 1', 'Formant 2', 'Formant 3']].values
    labels = gender[['Vowel Phoneme']].values.ravel()
    genders = gender[['Gender (M/F)']].values.ravel()

    # clustering is run for 2 different distance metrics, to identify the best approach
    for metric in [euclidean, cityblock]:
        # for every distance metric, the algorithm is run 5 times, with 5 different test and training sets.
        for x in range(5):
    
            # Split the data for the selected gender
            X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.25)
    
            # k is run for k in the range of 3 to 8, so that the optimal value of k is selected
            for k in range(3, 8):
                # initialize the classifier object
                classifier = neighbors.KNeighborsClassifier(n_neighbors=k, metric=metric)
                # train the classifier on our training data
                classifier.fit(X_train, y_train)
                # predict values for our test set
                y_pred = classifier.predict(X_test)
                # compute confusion matrices and F1 scores
                conf_matrix = confusion_matrix(y_test, y_pred)
                f1 = f1_score(y_test, y_pred, average='weighted')
                # append results to the dataframe
                results = results.append({'Gender': gender['Gender (M/F)'].values[0], 'Split Variation': x+1, 'k': k, 'Distance Metric': metric.__name__, 'Confusion Matrix': conf_matrix, 'F1 score': f1}, ignore_index=True)
    
# the results are finally exported to an excel sheet, which is easier to visualize in the report
results.to_excel('./resultsPerGender.xlsx')
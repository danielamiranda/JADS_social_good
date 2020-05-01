def fasttext_classification(usa, column_to_predict):
    import fasttext
    import pandas as pd
    import csv
    
    # validation: split the data into train and test set  
    from sklearn.model_selection import train_test_split
    usa_train, usa_test, y_train, y_test = train_test_split(usa, usa.tag, test_size=0.2)
    # save the data
    usa_train.to_csv(r'usa.train.txt', index=False, sep=' ', header=False, quoting=csv.QUOTE_NONE, quotechar="", escapechar=" ")
    usa_test.to_csv(r'usa.test.txt', index=False, sep=' ', header=False, quoting=csv.QUOTE_NONE, quotechar="", escapechar=" ")

    train_path = 'usa.train.txt'
    model = fasttext.train_supervised(input = train_path)
    results = model.test("usa.test.txt") # (number of samples, precision, recall)
    print(results)

    # build the model
    usa.to_csv(r'usa.txt', index=False, sep=' ', header=False, quoting=csv.QUOTE_NONE, quotechar="", escapechar=" ")
    train_path = 'usa.txt'
    model = fasttext.train_supervised(input = train_path)
    
    # predict in unknown data
    labels = []
    for i in range(len(column_to_predict)):
        label = model.predict(column_to_predict[i])
        labels.append(label)

    predictions = pd.DataFrame(labels)
    predictions.columns = ['label','probability']
    predictions['label'] = predictions['label'].astype(str)
    predictions['label'] = [s.replace("(",'').replace(')','').replace('__label__','').replace(',','').replace("'",'')  for s in predictions['label']]
    predictions['probability'] = predictions['probability'].astype(str)
    predictions['probability'] = [s.replace("[",'').replace(']','')  for s in predictions['probability']]
    predictions['probability'] = predictions['probability'].astype(float)
    return predictions
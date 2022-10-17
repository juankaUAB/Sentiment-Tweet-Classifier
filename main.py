from NaiveBayes import *

def main():
    plt.figure()
    dataset = pd.read_csv("FinalStemmedSentimentAnalysisDataset.csv",delimiter=";")
    dataset = dataset[["tweetText","sentimentLabel"]]
    dataset = dataset.dropna()
    
    
    print("#### APARTAT A ####")
    print("#### TRAIN-TEST SPLIT ####")
    test_one_model(dataset["tweetText"],dataset["sentimentLabel"],0,0.8)
    print("#### CROSS-VALIDATION ####")
    kfold(dataset.values, 5)
    print('')
    
    # Apartat B
    print("#### APARTAT B #####")
    particions = [0.2,0.3,0.4,0.5,0.6,0.7,0.8]
    for part in particions:
        print("--- Particio del ", part, " ---")
        x_t, x_v, y_t, y_v = train_test_split(dataset["tweetText"],dataset["sentimentLabel"], part)
        model = NaiveBayes()
        model.fit(x_t, y_t)
        print("Tamany del diccionari NEGATIU: ", len(model.probabilities_classN))
        print("Tamany del diccionari POSITIU: ", len(model.probabilities_classP))
        predictions = model.predict(x_v)
        print("Accuracy: ", accuracy_score(y_v, predictions))
    
    print("")
    tamany = [20000,50000,100000,200000,300000]
    for t in tamany:
        x_t, x_v, y_t, y_v = train_test_split(dataset["tweetText"],dataset["sentimentLabel"], 0.8)
        model = NaiveBayes()
        model.fit(x_t, y_t, t)
        print("Tamany del diccionari NEGATIU: ", len(model.probabilities_classN))
        print("Tamany del diccionari POSITIU: ", len(model.probabilities_classP))
        predictions = model.predict(x_v)
        print("Accuracy: ", accuracy_score(y_v, predictions))
    
    print("")
    for part in particions:
        print("--- Particio del ", part, " amb un tamany del diccionari de 100.000 paraules ---")
        x_t, x_v, y_t, y_v = train_test_split(dataset["tweetText"],dataset["sentimentLabel"], part)
        model = NaiveBayes()
        model.fit(x_t, y_t, 100000)
        print("Tamany del diccionari NEGATIU: ", len(model.probabilities_classN))
        print("Tamany del diccionari POSITIU: ", len(model.probabilities_classP))
        predictions = model.predict(x_v)
        print("Accuracy: ", accuracy_score(y_v, predictions))
      
    print("")
    print("#### APARTAT C ####")
    test_one_model(dataset["tweetText"],dataset["sentimentLabel"],1,0.8)
       
if __name__ == "__main__":
    main()
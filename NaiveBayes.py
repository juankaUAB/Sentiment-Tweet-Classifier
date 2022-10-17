import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import operator
from collections import Counter
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, precision_score, recall_score, roc_curve, auc, RocCurveDisplay, precision_recall_curve, PrecisionRecallDisplay

class NaiveBayes: #classe que implementa el model classificador, el Naive Bayes
    def __init__(self, alfa = 0): #constructor de la classe
        self.total_words = 0
        self.probability_classes = [0,0] # suposant sempre classes binaries
        self.probabilities_classP = Counter()
        self.probabilities_classN = Counter()
        self.words_classN = 0
        self.words_classP = 0
        self.smoothing = alfa
    
    def __limit_size_dict(self, size): # funcio per limitar el tamany del diccionari al que vulguem
        self.probabilities_classN = Counter(dict(self.probabilities_classN.most_common(size)))
        self.probabilities_classP = Counter(dict(self.probabilities_classP.most_common(size)))
        
    def __count_wordsEachClass(self, x, y, tamany_dict): #generar els diccionaris amb les frequencies de cada paraula diferent trobada
                                                         #i calcul de la probabilitat de cada classe (en aquest problema 0 i 1)
        x = x.reshape(-1)
        y = y.reshape(-1)
        for row, target in zip(x,y): #recorrer tots el tweets del conjunt de train
            if target == 0: #generar diccionari de la classe negativa
                words = row.split()
                self.probabilities_classN.update(words)
                self.words_classN += len(words)
                self.probability_classes[0] += 1
            else: #generar diccionari de la classe positiva
                words = row.split()
                self.probabilities_classP.update(words)
                self.words_classP += len(words)
                self.probability_classes[1] += 1
                
        #comptar total de paraules en tots els tweets i la probabilitat de cada classe
        self.total_words = self.words_classN + self.words_classP
        self.probability_classes[0] /= x.shape[0]
        self.probability_classes[1] /= x.shape[0]
        
        if tamany_dict != -1: #en cas que volguem limitar el tamany del diccionari a un especificat cridem aquesta funció
            self.__limit_size_dict(tamany_dict)
        
    
    def fit(self, x, y, tamany_dict = -1):
        self.__count_wordsEachClass(x, y, tamany_dict)
    
    def predict(self, x):
        x = x.reshape(-1)
        predictions = []
        if self.smoothing == 0: #no s'ha establert el metode de laplace smoothing. No s'utilitzen logaritmes pq log(0) no existeix
            for linea in x: #recorrer tots els tweets del conjunt de test per classificar-los
                words = linea.split()
                probP = 1
                probN = 1
                for word in words: #important: si la probabilitat es 0 parem de multiplicar per que surten warnings si continuem multiplicant
                    if probN != 0:
                        #aqui calculem la probabilitat de la paraula, ja que als diccionaris nomes tenim les frequencies
                        probN *= (self.probabilities_classN[word] + self.smoothing) / (self.words_classN + (self.smoothing * self.total_words))
                    if probP != 0:
                        probP *= (self.probabilities_classP[word] + self.smoothing) / (self.words_classP + (self.smoothing * self.total_words))
                if probN != 0:
                    probN *= self.probability_classes[0]
                if probP != 0:
                    probP *= self.probability_classes[1]
                predictions.append(np.argmax(np.array([probN,probP])))
        else: # s'ha establert un laplace smoothing i es fan logaritmes per evitar overflow amb decimals per la multiplicació de numeros amb molts decimals
            for linea in x:
                words = linea.split()
                probP = 0
                probN = 0
                for word in words:
                    probN += np.log((self.probabilities_classN[word] + self.smoothing) / (self.words_classN + (self.smoothing * self.total_words)))
                    probP += np.log((self.probabilities_classP[word] + self.smoothing) / (self.words_classP + (self.smoothing * self.total_words)))
                probN += np.log(self.probability_classes[0])
                probP += np.log(self.probability_classes[1])
                predictions.append(np.argmax(np.array([np.exp(probN),np.exp(probP)]))) #desfer la suma de logaritmes
        return predictions
                    
            
        

def train_test_split(x, y, part):
    union = pd.concat([x,y],axis=1)
    union_shuffle = union.loc[np.random.permutation(union.index)].reset_index(drop=True) #barregem les files del dataset
    x_shuffle = union_shuffle.iloc[:,0:-1]
    y_shuffle = union_shuffle.iloc[:,-1]
    x_train = x_shuffle.iloc[0:int(x_shuffle.shape[0]*part)]
    x_test = x_shuffle.iloc[int(x_shuffle.shape[0]*part):-1]
    y_train = y_shuffle.iloc[0:int(y_shuffle.shape[0]*part)]
    y_test = y_shuffle.iloc[int(y_shuffle.shape[0]*part):-1]
    return np.array(x_train.values,dtype=np.str), np.array(x_test.values,dtype=np.str), y_train.values, y_test.values


def kfold(dataset, n_particions,smoothing = 0): # estrategia de validació creuada per provar el nostre model
    """DIVIDIR DATASET EN N PARTICIONS"""
    new_dataset = dataset[np.random.permutation(len(dataset))]
    conjunts = []
    for inc,i in enumerate(range(0,new_dataset.shape[0],int(new_dataset.shape[0]/n_particions))):
        if inc < n_particions:
            seguent = (int(new_dataset.shape[0]/n_particions))*(inc+1)
            conjunts.append(new_dataset[i:seguent,:])
    """EXECUTAR CROSS-VALIDATION"""
    mitj = 0
    for i, test in enumerate(conjunts):
        conjunts.pop(i)
        train = np.concatenate(conjunts)
        
        model = NaiveBayes(smoothing)
        model.fit(np.array(train[:,0],dtype=str), np.array(train[:,-1],dtype=int))
        pred = model.predict(np.array(test[:,0],dtype=str))
        score = accuracy_score(np.array(test[:,-1],dtype=int), pred)
        mitj += score
        print("Predicció per al conjunt", i, ":", score)
        
        conjunts.insert(i, test)
    
    print("--------------------------")
    print("Resultat mitjà del cross-validation:", mitj / n_particions)
        
def test_one_model(x, y, smoothing, part):
    x_t, x_v, y_t, y_v = train_test_split(x,y, part)
    model = NaiveBayes(smoothing) #aqui es on es crea el model amb el valor que hem escollit pel laplace smoothing
    start = time.time()
    model.fit(x_t, y_t)
    predictions = model.predict(x_v)
    end = time.time()
    print("Elapsed time: ", end - start)
    print("Tamany diccionari de la classe positiva: ", len(model.probabilities_classP))
    print("Tamany diccionari de la classe negativa: ", len(model.probabilities_classN))
    print("Accuracy: ", accuracy_score(y_v, predictions))
    print("Precision: ", precision_score(y_v, predictions))
    print("Recall: ", recall_score(y_v, predictions))
    #ConfusionMatrix
    ConfusionMatrixDisplay.from_predictions(y_v, predictions)
    #RocCurve
    fpr, tpr, thresholds = roc_curve(y_v, predictions)
    roc_auc = auc(fpr, tpr)
    display = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc)
    display.plot()
    #PrecisionRecallCurve
    precision, recall, _ = precision_recall_curve(y_v, predictions)
    disp = PrecisionRecallDisplay(precision=precision, recall=recall)
    disp.plot()
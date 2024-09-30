import re
import math

# Stopword list
STOPWORDS = set(['the', 'is', 'at', 'which', 'on', 'and', 'a', 'an', 'this', 'that'])

# Clean and tokenize text, removing stopwords
def preprocessText(text):
    text = re.sub(r'[^\w\s]', '', text.lower())  # Remove punctuation and lowercase
    tokens = text.split()
    tokens = [word for word in tokens if word not in STOPWORDS]  # Remove stopwords
    return tokens

# Load data from file and preprocess it
def loadData(filePath, limit=None):
    with open(filePath, 'r') as f:
        lines = f.readlines()
        if limit:
            lines = lines[:limit]
    return [preprocessText(line) for line in lines]

# Split data into train, validation, and test sets
def splitData(posData, negData):
    posTrain, posVal, posTest = posData[:4000], posData[4000:4500], posData[4500:]
    negTrain, negVal, negTest = negData[:4000], negData[4000:4500], negData[4500:]
    
    trainData = posTrain + negTrain
    valData = posVal + negVal
    testData = posTest + negTest
    
    trainLabels = [1] * 4000 + [0] * 4000
    valLabels = [1] * 500 + [0] * 500
    testLabels = [1] * 831 + [0] * 831
    
    return (trainData, trainLabels), (valData, valLabels), (testData, testLabels)

# Train Naive Bayes with Laplace smoothing
def trainNaiveBayes(positiveData, negativeData):
    positiveWordCounts = {}
    negativeWordCounts = {}
    
    totalPositiveWords = 0
    totalNegativeWords = 0
    
    for review in positiveData:
        for word in review:
            if word in positiveWordCounts:
                positiveWordCounts[word] += 1
            else:
                positiveWordCounts[word] = 1
            totalPositiveWords += 1
    
    for review in negativeData:
        for word in review:
            if word in negativeWordCounts:
                negativeWordCounts[word] += 1
            else:
                negativeWordCounts[word] = 1
            totalNegativeWords += 1
    
    vocab = set(positiveWordCounts.keys()).union(negativeWordCounts.keys())
    vocabSize = len(vocab)
    
    priorPositive = len(positiveData) / (len(positiveData) + len(negativeData))
    priorNegative = len(negativeData) / (len(positiveData) + len(negativeData))
    
    return (positiveWordCounts, negativeWordCounts, totalPositiveWords, totalNegativeWords, vocabSize, priorPositive, priorNegative)

# Predict class using Naive Bayes
def predictClass(review, positiveWordCounts, negativeWordCounts, totalPositiveWords, totalNegativeWords, vocabSize, priorPositive, priorNegative):
    logProbPositive = math.log(priorPositive)
    logProbNegative = math.log(priorNegative)
    
    for word in review:
        positiveLikelihood = (positiveWordCounts.get(word, 0) + 1) / (totalPositiveWords + 1 * vocabSize)
        negativeLikelihood = (negativeWordCounts.get(word, 0) + 1) / (totalNegativeWords + 1 * vocabSize)
        
        logProbPositive += math.log(positiveLikelihood)
        logProbNegative += math.log(negativeLikelihood)
    
    return 1 if logProbPositive > logProbNegative else 0

# Evaluate accuracy on test set
def evaluateModel(testData, testLabels, positiveWordCounts, negativeWordCounts, totalPositiveWords, totalNegativeWords, vocabSize, priorPositive, priorNegative):
    correct = 0
    for i in range(len(testData)):
        prediction = predictClass(testData[i], positiveWordCounts, negativeWordCounts, totalPositiveWords, totalNegativeWords, vocabSize, priorPositive, priorNegative)
        if prediction == testLabels[i]:
            correct += 1
    return correct / len(testData)

# Calculate evaluation metrics
def evaluateWithMetrics(testData, testLabels, positiveWordCounts, negativeWordCounts, totalPositiveWords, totalNegativeWords, vocabSize, priorPositive, priorNegative):
    y_pred = []
    
    for review in testData:
        prediction = predictClass(review, positiveWordCounts, negativeWordCounts, totalPositiveWords, totalNegativeWords, vocabSize, priorPositive, priorNegative)
        y_pred.append(prediction)
    
    tp = tn = fp = fn = 0
    
    for actual, predicted in zip(testLabels, y_pred):
        if actual == 1 and predicted == 1:
            tp += 1
        elif actual == 0 and predicted == 0:
            tn += 1
        elif actual == 0 and predicted == 1:
            fp += 1
        elif actual == 1 and predicted == 0:
            fn += 1
    
    accuracy = (tp + tn) / len(testLabels)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return accuracy, precision, recall, f1, tp, tn, fp, fn

# Load datasets
positiveData = loadData('dataset/rt-polarity.pos')
negativeData = loadData('dataset/rt-polarity.neg')

# Split into train, validation, and test sets
(trainData, trainLabels), (valData, valLabels), (testData, testLabels) = splitData(positiveData, negativeData)

# Train Naive Bayes model
positiveTrain = trainData[:4000]
negativeTrain = trainData[4000:]
positiveWordCounts, negativeWordCounts, totalPositiveWords, totalNegativeWords, vocabSize, priorPositive, priorNegative = trainNaiveBayes(positiveTrain, negativeTrain)

# Evaluate model performance
accuracy, precision, recall, f1, tp, tn, fp, fn = evaluateWithMetrics(testData, testLabels, positiveWordCounts, negativeWordCounts, totalPositiveWords, totalNegativeWords, vocabSize, priorPositive, priorNegative)

print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")
print(f"True Positives: {tp}, True Negatives: {tn}, False Positives: {fp}, False Negatives: {fn}")
import math

class NaiveBayesDigitClassifier(object):
    def __init__(self):
        # Training sample count
        self.sampleCount = 0
        # Pixel grid size
        self.size = 28
        # digit label as key, Values are list of matrices
        self.trainingSet = dict()
         # Digit label as key,value = prior probability of digit
        self.labelProb = dict()
        # dictionary of digit- 3d matrix for each digit, stores a list of probabilities of ' ','+','#'
        self.pixelProbMap = dict()
        # Precision of each digit in test data
        self.labelPrecision = dict()

    # ' ' => 1, white pixel
    # '#' => 2, black pixel
    # '+' => 3, grey pixel
    def train(self, training_images, training_labels):
        # Open training image and label file
        imageHandle = open(training_images, 'r')
        labelHandle = open(training_labels, 'r')
        count = 0
        # Read self.size rows from images file for each label
        for label in labelHandle:
            count = count + 1
            label = label.rstrip('\n')
            sample = []
            for row in range(self.size):
                line = imageHandle.readline()
                line.rstrip('\n')
                row = list()
                for pixel in list(line):
                    if pixel == ' ':
                        v = 1
                    elif pixel == '#':
                        v = 2
                    elif pixel == '+':
                        v = 3
                    row.append(v)
                if row : row.pop()
                #Append each row to sample
                sample.append(row)
            # Add sample to training set dictionary for corresponding label as key
            if label in self.trainingSet:
                self.trainingSet[label].append(sample)
            else:
                self.trainingSet[label] = [sample]
        self.sampleCount = count
        self.trainFromSamples()


    def trainFromSamples(self):
        self.calcPriorLabelProbabilities()
        for label in self.trainingSet:
            self.calcPixelProbForLabel(label)

    def calcPriorLabelProbabilities(self):
        #Laplace smoothing is applied
        for label in self.trainingSet:
            # Compute Prior probability of each label, possible pixel values count = 3
            self.labelProb[label] = (len(self.trainingSet[label])+ 1)/ (float(self.sampleCount) + 3)


    def calcPixelProbForLabel(self, label):
        gridProb = [[x for x in range(self.size)] for y in range(self.size)]
        for row in range(self.size):
            for col in range(self.size):
                white = 0
                grey = 0
                black = 0
                for sample in self.trainingSet[label]:
                    if sample[row][col] == 1:
                        white += 1          #' '
                    elif sample[row][col] == 2:
                        black += 1          #'#'
                    elif sample[row][col] == 3:
                        grey += 1           #'+'
                pixelProb = list()
                """Laplace smoothing is applied while evaluating pixel probability"""
                pixelProb.append( (white + 1)/ (float(len(self.trainingSet[label])) + 3))  #' '
                pixelProb.append( (black + 1)/ (float(len(self.trainingSet[label]))+ 3))  #'#'
                pixelProb.append( (grey + 1)/ (float(len(self.trainingSet[label]))+ 3))   #'+'

                gridProb[row][col] = pixelProb
        # Add 3d matrix to dictionary with label as key
        self.pixelProbMap[label] = gridProb

    def testData(self, testImages, testLabel):
        imageHandle = open(testImages, 'r')
        labelHandle = open(testLabel, 'r')
        count = 0
        match = 0
        for label in labelHandle:
            count = count + 1
            label = label.rstrip('\n')
            sample = []
            for row in range(self.size):
                line = imageHandle.readline()
                line.rstrip('\n')
                row = list()
                for pixel in list(line):
                    if pixel == ' ':
                        v = 1
                    elif pixel == '#':
                        v = 2
                    elif pixel == '+':
                        v = 3
                    row.append(v)
                if row : row.pop()
                sample.append(row)

            #applying  naive bayes here
            labelValPair = []
            if label in self.labelPrecision:
                labelValPair = self.labelPrecision[label]
                labelValPair[0] += 1
            else:
                self.labelPrecision[label]  = [0,0]
                labelValPair = self.labelPrecision[label]
                labelValPair[0] += 1
            # If label match given test label, increment count
            if self.testSample(label, sample):
                match += 1
                labelValPair[1] += 1


        print "Accuracy of Naive Bayes classifier = " + str(match * 100/ float(count))+ "%"
        print "Digit      Precision"
        for digit in sorted(self.labelPrecision):
            lvPair = self.labelPrecision[digit]
            if lvPair != 0:
                print " " + digit + "\t\t\t"+ str("%.2f" % (lvPair[1] * 100/ float(lvPair[0]))) + "%"
            else:
                print " " + digit +"\t\t\t" + "---"


    # Test sample using Naive Bayes classifier, find max hypothesis
    def testSample(self, test_label, sample):
        vmap = -float('inf')
        out_label = '0'
        for train_label in self.pixelProbMap:
            trainedPixelGrid = self.pixelProbMap[train_label]
            v_j = 0.0
            for row in range(self.size):
                for col in range(self.size):
                    val = sample[row][col]
                    prob = trainedPixelGrid[row][col][val-1]
                    v_j += math.log(prob)
            v_j += math.log(self.labelProb[train_label])
            if v_j > vmap:
                vmap = v_j
                out_label = train_label

        return (out_label == test_label)


def main():
    print "Naive Bayes Classifier for Handwritten digits!!!"
    classifier = NaiveBayesDigitClassifier()
    print "Training on input images..."
    classifier.train('trainingimages.txt', 'traininglabels.txt')
    print "Training complete."
    print "Classifying test data..."
    classifier.testData('testimages.txt', 'testlabels.txt')


if __name__ == '__main__':
    main()

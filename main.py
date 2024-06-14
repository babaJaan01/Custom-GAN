# run this

import csv
from model import iterations, deltaX

def percentage(addedUp, howMany): # getting %
    first = (1 - abs(addedUp - howMany) / howMany) * 100
    print(f"{round(first, 2)}% accurate to the original data")

def addAccuracies(filePath, columnName): # just adding up all the accuracy values
    total = 0.0
    with open(filePath, 'r') as file:
        csvReader = csv.DictReader(file)
        for row in csvReader:
            total += float(row[columnName])
    return total

filePath = r'C:\Code\GANCode\logs.csv'
columnName = ' Accuracy'
addedUp = addAccuracies(filePath, columnName)
howMany = ((iterations - 1) / 10) * 0.5 # this is because in the csv it shows data every 10 iterations

print(f'Total added up in Accuracy is: {addedUp}')

percentage(addedUp, howMany)
from nltk.tokenize import sent_tokenize, word_tokenize
from operator import itemgetter
from collections import Counter
import re
import string
import json
import numpy as np

def main():
    file = open("languagesAll.txt", 'r')
    testResult = []
    totalLines = 0
    totalCorrect = 0
    if (file):
        langData = file.read()
        langs = langData.split("\n")
        setText('x_train.txt')
        for lang in langs:
            trainProfile(lang, 'y_train.txt', jsonfile = lang + '.json', save_to_json = True)
        setText('x_test.txt')
        for lang in langs:
            testResult.append(testProfile(lang, 'y_test.txt', langs))
        for langResult in testResult:
            #print(langResult)
            print("{}:\n\tAccuracy: {} / {} = {}%".format(langResult[0], langResult[1], langResult[2], (langResult[1] / langResult[2]) * 100))
            totalLines += langResult[2]
            totalCorrect += langResult[1]
        print("Average Accuracy: {} / {} = {}%".format(totalCorrect, totalLines, (totalCorrect / totalLines) * 100))
    else:
        print("File Error")

# Reads in all text from the training set into an array
def setText(filename):
    global text
    file = open(filename, encoding = "utf8")
    if (file):
        textData = file.read()
        text = textData.split("\n")
    else:
        print("File Error")

# Trains the model for the given language and saves to json
def trainProfile(lang, trainFilename, jsonfile = None, save_to_json = False):
    # Get set of languages' text into array
    textArray = getText(lang, trainFilename)
    # Tokenize text and gives it a model
    profile = {}
    for line in textArray:
        line = tokenizeString(line)
        textProfile = setProfile(line)
        # Update given languages' profile from the text profile
        if (profile):
            for item in textProfile:
                if item in profile:
                    profile[item] += textProfile[item]
                else:
                    profile[item] = textProfile[item]
        else:
            profile = textProfile
    profile = profileSort(profile)
    # Save to json
    if save_to_json:
        profileFile = jsonfile
        with open(profileFile, 'w') as outfile:
            json.dump(profile, outfile, indent = 4, sort_keys = True)
        print("Saved ", profileFile)

# Stores all text lines for a specified language into an array
def getText(lang, filename):
    index = 0
    textArray = []
    file = open(filename, 'r')
    if (file):
        textData = file.read()
        langs = textData.split("\n")
    for language in langs:
        if lang == language:
            textArray.append(text[index])
        index += 1
    return textArray

# Removes all numbers and punctuation from line of text
def tokenizeString(line):
    line = ''.join([i for i in line if not i.isdigit()])
    line = re.sub(r'[^\w\s]', '', line)
    line = word_tokenize(line)
    return line

# Creates model for a given line of text
def setProfile(line):
    profile = Counter()
    # Iterates through all words in a line of text
    for word in line:
        word = " " + word + " "
        # Creates ngram profiles (1 - 5 gram)
        for n in range(1, 6):
            # Get the key value pair for the ngram
            nProfile = nGram(word, n)
            # Update profile
            if (profile):
                for item in nProfile:
                    if item in profile:
                        profile[item] += nProfile[item]
                    else:
                        profile[item] = nProfile[item]
            else:
                profile = nProfile
    # Deletes ' ' gram
    if " " in profile:
        del profile[" "]
    return profile

# Sorts the profile based on the frequency of the ngram
def profileSort(profile):
    # Sort from high frequency to low
    profile = sorted(profile.items(), key = itemgetter(1), reverse = True)
    # Generate key array from the sorted profile
    profile = [x[0] for x in profile]
    # Keep only the top 300 keys (ngrams)
    profile = profile[:300]
    return profile

# Generate all ngrams for a given word
def nGram(word, n):
    # Create an empty profile
    profile = {}
    i = 0
    # Iteration through the word
    while (i < len(word) - n + 1):
        gram = ''
        j = 0
        # Current position in the word
        counter = i
        # Generate the ngram given the current position in the word
        while (j < n):
            gram += word[counter]
            j += 1
            counter += 1
        if gram in profile:
            profile[gram] += 1
        else:
            profile[gram] = 1
        i += 1
    return profile

# Determines accuracy of the language classification
def testProfile(language, langFile, languages):
    textArray = getText(language, langFile)
    lineCount = 0
    score = 0
    profile = {}
    result = []
    # Iterates through all lines of a given language and determines correctness of prediction
    for line in textArray:
        line = tokenizeString(line)
        profile = setProfile(line)
        profile = profileSort(profile)
        classification = calcLanguage(profile, languages)
        if classification == language:
            score += 1
        lineCount += 1
    result.append(language)
    result.append(score)
    result.append(lineCount)
    return result

# Determines language of a line by selecting the shortest distance between profile comparison
def calcLanguage(profile, languages):
    closestLang = ""
    shortestDist = 0
    for lang in languages:
        dist = 0
        dist = distance(profile, lang)
        if shortestDist == 0 or dist < shortestDist:
            shortestDist = dist
            closestLang = lang
    return closestLang

# Calculates the distance of a given language
def distance(profile, language):
    # Open .json file of given language
    with open(language + ".json") as dataFile:
        data = json.load(dataFile)
    p = np.asarray(data)
    p = p.tolist()
    i = 0
    distance = 0
    # Gets the distance between the given languages profile and the lines profile
    while i < len(profile):
        if profile[i] in p:
            j = p.index(profile[i])
            if i > j:
                distance = distance + i - j
            elif j > i:
                distance = distance + j - i
        else:
            distance += 300
        i += 1
    return distance

main()
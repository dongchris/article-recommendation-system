import sys
import re
import string
import os
import numpy as np
import codecs

# From scikit learn that got words from:
# http://ir.dcs.gla.ac.uk/resources/linguistic_utils/stop_words
ENGLISH_STOP_WORDS = frozenset([
    "a", "about", "above", "across", "after", "afterwards", "again", "against",
    "all", "almost", "alone", "along", "already", "also", "although", "always",
    "am", "among", "amongst", "amoungst", "amount", "an", "and", "another",
    "any", "anyhow", "anyone", "anything", "anyway", "anywhere", "are",
    "around", "as", "at", "back", "be", "became", "because", "become",
    "becomes", "becoming", "been", "before", "beforehand", "behind", "being",
    "below", "beside", "besides", "between", "beyond", "bill", "both",
    "bottom", "but", "by", "call", "can", "cannot", "cant", "co", "con",
    "could", "couldnt", "cry", "de", "describe", "detail", "do", "done",
    "down", "due", "during", "each", "eg", "eight", "either", "eleven", "else",
    "elsewhere", "empty", "enough", "etc", "even", "ever", "every", "everyone",
    "everything", "everywhere", "except", "few", "fifteen", "fifty", "fill",
    "find", "fire", "first", "five", "for", "former", "formerly", "forty",
    "found", "four", "from", "front", "full", "further", "get", "give", "go",
    "had", "has", "hasnt", "have", "he", "hence", "her", "here", "hereafter",
    "hereby", "herein", "hereupon", "hers", "herself", "him", "himself", "his",
    "how", "however", "hundred", "i", "ie", "if", "in", "inc", "indeed",
    "interest", "into", "is", "it", "its", "itself", "keep", "last", "latter",
    "latterly", "least", "less", "ltd", "made", "many", "may", "me",
    "meanwhile", "might", "mill", "mine", "more", "moreover", "most", "mostly",
    "move", "much", "must", "my", "myself", "name", "namely", "neither",
    "never", "nevertheless", "next", "nine", "no", "nobody", "none", "noone",
    "nor", "not", "nothing", "now", "nowhere", "of", "off", "often", "on",
    "once", "one", "only", "onto", "or", "other", "others", "otherwise", "our",
    "ours", "ourselves", "out", "over", "own", "part", "per", "perhaps",
    "please", "put", "rather", "re", "same", "see", "seem", "seemed",
    "seeming", "seems", "serious", "several", "she", "should", "show", "side",
    "since", "sincere", "six", "sixty", "so", "some", "somehow", "someone",
    "something", "sometime", "sometimes", "somewhere", "still", "such",
    "system", "take", "ten", "than", "that", "the", "their", "them",
    "themselves", "then", "thence", "there", "thereafter", "thereby",
    "therefore", "therein", "thereupon", "these", "they", "thick", "thin",
    "third", "this", "those", "though", "three", "through", "throughout",
    "thru", "thus", "to", "together", "too", "top", "toward", "towards",
    "twelve", "twenty", "two", "un", "under", "until", "up", "upon", "us",
    "very", "via", "was", "we", "well", "were", "what", "whatever", "when",
    "whence", "whenever", "where", "whereafter", "whereas", "whereby",
    "wherein", "whereupon", "wherever", "whether", "which", "while", "whither",
    "who", "whoever", "whole", "whom", "whose", "why", "will", "with",
    "within", "without", "would", "yet", "you", "your", "yours", "yourself",
    "yourselves"])


def load_glove(filename):
    """
    Read all lines from the indicated file and return a dictionary
    mapping word:vector where vectors are of numpy `array` type.
    GloVe file lines are of the form:

    the 0.418 0.24968 -0.41242 0.1217 ...

    So split each line on spaces into a list; the first element is the word
    and the remaining elements represent factor components. The length of the vector
    should not matter; read vectors of any length.
    """

    f = open(filename)
    lines = f.readlines()
    f.close()

    d = {}

    for line in lines:
        words = line.split(' ')
        word = words[0]
        vector = np.array(words[1:], dtype = 'float')
        d[word] = vector

    return d


def filelist(root):
    """Return a fully-qualified list of filenames under root directory"""
    allfiles = []
    for path, subdirs, files in os.walk(root):
        for name in files:
            allfiles.append(os.path.join(path, name))
    return allfiles


def get_text(filename):
    """
    Load and return the text of a text file, assuming latin-1 encoding as that
    is what the BBC corpus uses.  Use codecs.open() function not open().
    """
    f = codecs.open(filename, encoding='latin-1', mode='r')
    s = f.read()
    f.close()
    return s


def words(text):
    """
    Given a string, return a list of words normalized as follows.
    Split the string to make words first by using regex compile() function
    and string.punctuation + '0-9\\r\\t\\n]' to replace all those
    char with a space character.
    Split on space to get word list.
    Ignore words < 3 char long.
    Lowercase all words
    Remove English stop words
    """

    regex = re.compile('[' + re.escape(string.punctuation) + '0-9\\r\\t\\n]')
    nopunct = regex.sub(' ', text)  # delete stuff but leave at least a space to avoid clumping together
    words = nopunct.split(' ' )
    words = [w for w in words if len(w) >= 3]  # ignore a, an, to, at, be, ...
    words = [w.lower() for w in words]
    words = [word for word in words if word not in ENGLISH_STOP_WORDS]
    # print words
    return words

def load_articles(articles_dirname, gloves):
    """
    Load all .txt files under articles_dirname and return a table (list of lists/tuples)
    where each record is a list of:

      [filename, title, article-text-minus-title, wordvec-centroid-for-article-text]

    We use gloves parameter to compute the word vectors and centroid.

    The filename is stripped of the prefix of the articles_dirname pulled in as
    script parameter sys.argv[2]. E.g., filename will be "business/223.txt"
    """

    # if file name does not end in /, then add it to the end of the file name
    if articles_dirname[-1]!='/':
        articles_dirname = articles_dirname + '/'

    category = os.walk(articles_dirname).next()[1]
    
    allfiles = []

    # get list of fully-qualified filenames
    for i in range(len(category)):
        s = articles_dirname + '%s' % category[i]
        allfiles = allfiles + filelist(s)

    condenselist = []

    # strip prefix
    for i in range(len(allfiles)):
        path = allfiles[i]
        dirname = os.path.basename(os.path.dirname(path))
        textname = os.path.basename(path)
        condenselist.append(dirname + '/' + textname)

    # get title
    # text in article without the title
    title = []
    article_without_title = []

    for i in range(len(allfiles)):
        f = open(allfiles[i])
        text = f.read()
        temp = text.split('\n')
        title.append(temp[0])
        text = "\n".join(temp[1:])
        article_without_title.append(text)
        f.close()
    
    # list to store centroids

    centroidlist = []

    for i in range(len(article_without_title)):
        centroidlist.append(doc2vec(article_without_title[i], gloves))

    # combine everything

    combinelist = []

    for i in range(len(article_without_title)):

        combinelist.append([condenselist[i], title[i], article_without_title[i],
            centroidlist[i]])
    return combinelist

# take care of encoding problems
reload(sys)
sys.setdefaultencoding('utf8')

def doc2vec(text, gloves):
    """
    Return the word vector centroid for the text. Sum the word vectors
    for each word and then divide by the number of words. Ignore words
    not in gloves.
    """

    # for i in range(len(articles[0])):
    #     if articles[2][i]==text:
    #         return articles[3][i]

    
    articlewords = words(text)
    wordvec = [gloves[word] for word in articlewords if word in gloves]

    

    lengthvec = len(wordvec)

    centroid = [sum(x)/lengthvec for x in zip(*wordvec)]

    return centroid


def distances(article, articles): # article = business/001.txt
    """
    Compute the euclidean distance from article to every other article and return
    a list of (distance, a) tuples for all a in articles. The article is one
    of the elements (tuple) from the articles list.
    """
    index = -99999
    for i in range(len(articles)):
        if articles[i][0] == article:
            comparewithvec = articles[i][3]
            index = i
            break

    comparewithvec = np.array(comparewithvec)
    
    articlescopy = articles

    distancelist = []

    for i in range(len(articlescopy)):
        dist = np.linalg.norm(np.array(articlescopy[i][3]) - comparewithvec)

        distancelist.append((dist, articlescopy[i]))
    distancelist.pop(index)
    return distancelist
    
    

def recommended(article, articles, n):
    """
    Return a list of the n articles (records with filename, title, etc...)
    closest to article's word vector centroid. The article is one of the elements
    (tuple) from the articles list.
    """

    d = distances(article, articles)
    
    toparticle = np.argsort([d[i][0] for i in range(len(d))])

    return [d[toparticle[i]] for i in range(n)]



    

from sklearn.feature_extraction import text
from numpy import log

def tf(t, d, mode="raw"):
    """ The Term Frequency 'tf' calculates how often a term 't'
        occurs in a document 'd'.  ('d': document index)
        If t_in_d =  Number of times a term t appears in a document d
        and no_terms_d = Total number of terms in the document,
        tf(t, d) = t_in_d / no_terms_d

    """

    if t in vectorizer.vocabulary_:
        word_ind = vectorizer.vocabulary_[t]
        t_occurences = da[d, word_ind]  # 'd' is the document index
    else:
        t_occurences = 0
    if mode == "raw":
        result = t_occurences
    elif mode == "length":
        all_terms = (da[d] > 0).sum()  # calculate number of different terms in d
        result = t_occurences / all_terms
    elif mode == "log":
        result = log(1 + t_occurences)
    elif mode == "augfreq":
        result = 0.5 + 0.5 * t_occurences / da[d].max()

    return result

corpus = [
            "It does not matter what you are doing, just do it!",
            "Would you work if you won the lottery?",
            "You like Python, he likes Python, we like Python, everybody loves Python!",
            "You said: 'I wish I were a Python programmer'",
            "You can stay here, if you want to. I would, if I were you."
         ]

vectorizer = text.CountVectorizer()
print(vectorizer)
vectorizer.fit(corpus)

print("Vocabulary: ", vectorizer.vocabulary_)
print(vectorizer.get_feature_names_out())
print(list(vectorizer.vocabulary_.keys()))

token_count_matrix = vectorizer.transform(corpus)
print(token_count_matrix)

dense_tcm = token_count_matrix.toarray()
print(dense_tcm)

feature_names = vectorizer.get_feature_names_out()
for el in vectorizer.vocabulary_:
    print(el, end=(", "))

#import pandas as pd

#PD = pd.DataFrame(data=dense_tcm,
#             index=['corpus_0', 'corpus_1', 'corpus_2', 'corpus_3', 'corpus_4'],
#             columns=vectorizer.get_feature_names_out())

#print(PD)

word = "you"
i = 1
j = vectorizer.vocabulary_[word]
print("number of times '" + word + "' occurs in:")
for i in range(len(corpus)):
     print("    '" + corpus[i] + "': " + str(dense_tcm[i][j]))

# txt = "That is the question and it is nobler in the mind."

# print("\n\n", txt)
# print(vectorizer.transform([txt]).toarray())

tf_idf = text.TfidfTransformer()
tf_idf.fit(token_count_matrix)

print(tf_idf.idf_)
print(tf_idf.idf_[vectorizer.vocabulary_['python']])

word_weight_list = list(zip(vectorizer.get_feature_names_out(), tf_idf.idf_))

word_weight_list.sort(key=lambda x:x[1])  # sort list by the weights (2nd component)
for word, idf_weight in word_weight_list:
    print(f"{word:15s}: {idf_weight:4.3f}")

da = vectorizer.transform(corpus).toarray()

print("   raw    length log   augmented freq")
for term in ['matter', 'python', 'would']:
    for docu_index in range(len(corpus)):
        d = corpus[docu_index]
        print(f"\n'{term}' in '{d}''")
        for mode in ['raw', 'length', 'log', 'augfreq']:
            x = tf(term, docu_index, mode=mode)
            print(f"{x:7.2f}", end="")


n = len(corpus)

def df(t):
    """ df(t) is the document frequency of t; the document frequency is
        the number of documents  in the document set that contain the term t. """

    word_ind = vectorizer.vocabulary_[t]

    tf_in_docus = da[:, word_ind]  # vector with the freqencies of word_ind in all docus
    existence_in_docus = tf_in_docus > 0  # binary vector, existence of word in docus
    return existence_in_docus.sum()


# df("would", vectorizer)

def idf(t, smooth_idf=True):
    """ idf """
    if smooth_idf:
        return log((1 + n) / (1 + df(t))) + 1
    else:
        return log(n / df(t)) + 1


def tf_idf(t, d):
    return idf(t) * tf(t, d)


res_idf = []
for word in vectorizer.get_feature_names_out():
    tf_docus = []
    res_idf.append([word, idf(word)])

res_idf.sort(key=lambda x: x[1])
for item in res_idf:
    print(item)

for word, word_index in vectorizer.vocabulary_.items():
    print(f"\n{word:12s}: ", end="")
    for d_index in range(len(corpus)):
        print(f"{d_index:1d} {tf_idf(word, d_index):3.2f}, ",  end="" )
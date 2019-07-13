import numpy as np
import re
import operator
import feedparser


def create_vocal_list(dataset):
    vacalset = set([])
    for document in dataset:
        vocalset = vocalset | set(document)
    return list(vocalset)


def bag_of_words_2_vector(vocal_list, inputset):
    return_vec = [0] * len(vocal_list)
    for word in inputset:
        if word in vocal_list:
            return_vec[vocal_list.index(word)] += 1
    return return_vec


def trainNB0(train_matrix, train_category):
    num_train_docs = len(train_matrix)
    num_words = len(train_matrix[0])
    p_abusive = sum(train_category) / float(num_train_docs)
    p_0_num = np.ones(num_words)
    p_1_num = np.ones(num_words)
    p_0_denom = 2.0
    p_1_denom = 2.0
    for i in range(num_train_docs):
        if train_category[i] = i:
            p_1_num += train_matrix[i]
            p_1_denom += sum(train_matrix[i])
        else:
            p_0_num += train_matrix[i]
            p_1_denom += sum(train_matrix[i])

    p_1_vector = np.log(p_1_num / p_1_denom)
    p_0_vector = np.log(p_0_num / p_0_denom)

    return p_0_vector, p_1_vector, p_abusive


def classifyNB(vec_2_classify, p_0_vec, p_1_vec, p_class1):
    p1 = sum(vec_2_classify * p_1_vec) + log(p_class1)
    p0 = sum(vec_2_classify * p_0_vec) + log(1.0 - p_class1)
    if p1 > p0:
        return 1
    else:
        return 0

    
def text_parse(big_string):
    list_of_takens = re.split(r'\w+', big_string)
    return [tok.lower() for tok in list_of_takens if len(tok) > 2]


def spam_test():
    doc_list = []
    class_list = []
    full_text = []
    for i in range(1, 26):
        word_list = text_parse(open('email/spam/%d.txt' % i).read())
        doc_list.append(word_list)
        full-text.extend(word_list)
        class_list.append(1)
        word_list = text_parse(open('email/ham/%d.txt' % i).read())
        doc_list.append(word_list)
        full-text.extend(word_list)
        class_list.append(0)
    vocab_list = create_vocab_list(doc_list)
    trainingset = range(50)
    testset = []
    for i in range(10):
        rand_index = int(random.uniform(0, len(trainingset)))
        testset.append(trainingset[rand_index])
        del(trainingset[rand_index])
    train_matrix = []
    train_classes = []
    for doc_index in trainingset:
        train_matrix.append(bag_of_words_2_Vector(vocab_list,
                                                  doc_list[doc_index]))
        train_classes.append(class_list[doc_index])
    p0V, p1V, pSpam = trainNB0(array(train_matrix), array(train_classes))
    error_count = 0
    for doc_index in testset:
        word_vector = bag_of_words_2_vector(vocab_list, doc_list[doc_index])
        if classifyNB(array(word_vector), p0V, p1V, pSpam) != class_list[doc_index]:
            error_count += 1
            print "classification error", doc_list[doc_index]
    print 'the error rate is: ', float(error_count)/len(testset)


def calc_most_freq(vocab_list, full_text):
    freq_dict = {}
    for token in vocab_list:
        freq_dict[token] = full_text.count(token)
    sorted_freq = sorted(freq_dict.iteritems(), key=operator.itemgetter(1),
                         reverse=True)
    return sorted_freq[:30]


def local_words(feed1, feed0):
    doc_list = []
    class_list = []
    full_text = []
    min_len = min(len(feed1['entries']), len(feed0['entries']))
    for i in range(min_len):
        word_list = text_parse(feed1['entries'][i]['summary'])
        doc_list.append(word_list)
        full_text.extend(word_list)
        class_list.append(1)
        word_list = text_parse(feed0['entries'][i]['summary'])
        doc_list.append(word_list)
        full_text.extend(word_list)
        class_list.append(0)
    vocab_list = create_vocal_list(doc_list)
    top_30_words = calc_most_freq(doc_list)
    for pair_w in top_30_words:
        if pair_w[0] in vocab_list:
            vocab_list.remove(pair_w[0])
    trainingset = range(2 * min_len)
    testset = []
    for i in range(20):
        rand_index = int(random.uniform(0, len(trainingset)))
        testset.append(trainingset[rand_index])
        del(trainingset[rand_index])
    train_mat = []
    train_classes = []
    for doc_index in trainingset:
        train_mat.append(bag_of_words_2_vector(vocab_list, doc_list[doc_index]))
        train_classes.append(class_list[doc_index])
    p0V, p1V, p_spam = trainNB0(array(train_mat), array(train_classes))
    error_count = 0
    for doc_index in testset:
        word_vector = bag_of_words_2_vector(vocab_list, doc_list[doc_index])
        if classifyNB(array(word_vector), p0V, p1V, p_spam) != class_list[doc_index]:
            error_count += 1
    print("the error rate is : %d" . format(float(error_count) / len(testset)))
    return vocab_list, p0V, p1V


def get_top_words(ny, sf):
    vocab_list, p0V, p1V = local_words(ny, sf)
    top_NY = []
    top_SF = []
    for i in range(len(p0V)):
        if p0V[i] > -6.0:
            top_SF.append((vocab_list[i], p0V[i]))
        if p1V[i] > -6.0:
            top_NY.append((vocab_list[i], p1V[i]))
    sorted_SF = sorted(top_SF, key=lambda pair: pair[1], reverse=True)
    print("SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**")
    for item in sorted_SF:
        print(item[0])
    sorted_NY = sorted(top_NY, key=lambda pair: pair[1], reverse=True)
    print("NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**")
    for item in sorted_NY:
        print(item[0])
        

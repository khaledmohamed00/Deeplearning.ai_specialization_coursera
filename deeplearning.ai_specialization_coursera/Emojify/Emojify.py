import numpy as np
from emo_utils import *
import matplotlib.pyplot as plt
import emoji


X_train, Y_train = read_csv('data/train_emoji.csv')
X_test, Y_test = read_csv('data/tesss.csv')
maxLen = len(max(X_train, key=len).split())

Y_oh_train = convert_to_one_hot(Y_train, C = 5)
Y_oh_test = convert_to_one_hot(Y_test, C = 5)

word_to_index, index_to_word, word_to_vec_map = read_glove_vecs('data/glove.6B.50d.txt')




def sentence_to_avg(sentence, word_to_vec_map):
    
    words=sentence.lower().split()
    
    av_vector=np.zeros([1,50])
    
    for word in words:
        word_vect=word_to_vec_map[word].reshape([1,50])
        av_vector+=word_vect
        
    av_vector/=len(words) 
    
    return av_vector

def model(X, Y, word_to_vec_map, learning_rate = 0.01, num_iterations = 400):
    m = Y.shape[0]                          # number of training examples
    n_y = 5                                 # number of classes  
    n_h = 50                                # dimensions of the GloVe vectors 
    
    Y_hot = convert_to_one_hot(Y, C = n_y)
    
    weight=np.random.randn(n_h,n_y)/np.sqrt(n_h)
    biase = np.zeros((1,n_y))
    
    for itr in range(num_iterations):
        for i in range(m):
            input_sentence_vector=sentence_to_avg(X[i], word_to_vec_map)
            input_sentence_vector=input_sentence_vector.reshape([1,50])
            z=input_sentence_vector.dot(weight)+biase
            a=softmax(z)
            
            cost = -np.sum(Y_hot * np.log(a))
            
            dz=a-Y_hot[i]
            dw=input_sentence_vector.T.dot(dz)
            db=dz
            
            weight-=learning_rate*dw
            biase-=learning_rate*db
            
        if itr % 100 == 0:
            print("Epoch: " + str(itr) + " --- cost = " + str(cost))
            pred = predict(X, Y, weight, biase, word_to_vec_map)

    return pred, weight, biase   
            
            
            
            
pred, W, b = model(X_train, Y_train, word_to_vec_map)
print(pred)            
print("Training set:")
pred_train = predict(X_train, Y_train, W, b, word_to_vec_map)
print('Test set:')
pred_test = predict(X_test, Y_test, W, b, word_to_vec_map)    

X_my_sentences = np.array(["This movie is not good and not enjoyable","fuck you", "funny lol", "lets play with a ball", "food is ready", "not feeling happy"])
Y_my_labels = np.array([[0], [0], [2], [1], [4],[3]])

pred = predict(X_my_sentences, Y_my_labels , W, b, word_to_vec_map)
print_predictions(X_my_sentences, pred)

print(Y_test.shape)
print('           '+ label_to_emoji(0)+ '    ' + label_to_emoji(1) + '    ' +  label_to_emoji(2)+ '    ' + label_to_emoji(3)+'   ' + label_to_emoji(4))
print(pd.crosstab(Y_test, pred_test.reshape(56,), rownames=['Actual'], colnames=['Predicted'], margins=True))
plot_confusion_matrix(Y_test, pred_test)
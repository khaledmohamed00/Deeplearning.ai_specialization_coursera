import numpy as np
from w2v_utils import *


words, word_to_vec_map = read_glove_vecs('data/glove.6B.50d.txt')


def cosine_similarity(u, v):
    u=u.reshape(1,u.shape[0])
    v=v.reshape(1,v.shape[0])
    
    dot_product=u.dot(v.T)
    
    norm_u=np.sqrt(np.sum(u**2))
    norm_v=np.sqrt(np.sum(v**2))
    
    cosine_similar= dot_product/(norm_u*norm_v)
    
    return cosine_similar

def complete_analogy(word_a, word_b, word_c, word_to_vec_map):
    
    vec_a=word_to_vec_map[word_a]
    vec_b=word_to_vec_map[word_b]
    vec_c=word_to_vec_map[word_c]
    
    words = word_to_vec_map.keys()
    best_word = None
    cosine_similarity_max=-9999
    for word in words:
        vec_word=word_to_vec_map[word]
        u=vec_b-vec_a
        v=vec_word-vec_c 
        similarity=cosine_similarity(u, v)
        if similarity>cosine_similarity_max:
           cosine_similarity_max= similarity
           best_word=word
           
    return best_word   
           
           
father = word_to_vec_map["father"]
mother = word_to_vec_map["mother"]
ball = word_to_vec_map["ball"]
crocodile = word_to_vec_map["crocodile"]
france = word_to_vec_map["france"]
italy = word_to_vec_map["italy"]
paris = word_to_vec_map["paris"]
rome = word_to_vec_map["rome"]

print("cosine_similarity(father, mother) = ", cosine_similarity(father, mother))
print("cosine_similarity(ball, crocodile) = ",cosine_similarity(ball, crocodile))
print("cosine_similarity(france - paris, rome - italy) = ",cosine_similarity(france - paris, rome - italy))

triads_to_try = [('italy', 'italian', 'spain'), ('india', 'delhi', 'japan'), ('man', 'woman', 'boy'), ('small', 'smaller', 'large')]
for triad in triads_to_try:
    print ('{} -> {} :: {} -> {}'.format( *triad, complete_analogy(*triad,word_to_vec_map)))
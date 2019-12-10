#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from itertools import groupby
import itertools
import timeit


# In[2]:


def load_file(path):
    train = []
    with open(path) as fp:
        for empty, line in groupby(fp, lambda x: x.startswith('\n')):
            if not empty:
                train.append(np.array([[str(x) for x in d.split()] for d in line if len(d.strip())]))
    print(train[0])
    return train
    


# In[8]:


def smooth(data, k):
  # returns list and count of words to be replaced with #UNK#
  df = pd.DataFrame(np.concatenate(data)) 
  df.columns = ['x','y']
  
  word_count = df['x'].value_counts()
  replace = word_count[word_count < k].index.tolist() # list of words to replace
  replace_count = word_count[word_count < k].sum() # total number of replaced words

  return replace, replace_count


# In[9]:


def init_values(data):
  # with smoothing
  # returns list of x, dictionary of label counts, empty dataframes for model params au,v & bu(o)
  start = timeit.default_timer()

  column = []
  rows = ['START']
  x = ['#UNK#']
  label_count = {}
  slist, scount = smooth(data, 3)
  count = 0

  for sequence in data:
    for word in sequence:
      # smoothing, modify training set
      if word[0] in slist:
        word[0] = "#UNK#"
        count += 1

      # get u, v
      if word[1] not in column:
        column.append(word[1])
        rows.append(word[1])
        label_count[word[1]] = 1
      else:
        label_count[word[1]] += 1

      # get x
      if word[0] not in x:
        x.append(word[0])

  column.append('STOP')
  # create dataframes to store model params
  dfA = pd.DataFrame(columns = column, index=rows)
  dfB = pd.DataFrame(columns = x, index=list(label_count))

  print(count)
  print(scount)

  stop = timeit.default_timer()
  print('Time: ', stop - start) 

  return x, label_count, dfA, dfB



# In[10]:


def emission(dfB, data, labels):
  # returns emission parameters
  dfB.fillna(0., inplace=True)

  for sequence in data:
    for word in sequence:
      label_total = labels[word[1]]
      dfB[word[0]][word[1]] += 1/label_total
  
  return dfB


# In[12]:


def sentiment(emis):
  # returns dictionary of words and corresponding y*
  
  maximum = emis.idxmax(axis = 0) # get maximum of each column
  tag = maximum.to_dict()

  return tag


# In[15]:


def transition(dfA, data, labels):
  # returns transition parameters
  start = timeit.default_timer()
  
  dfA.fillna(0., inplace=True)

  for i in range(len(data)):
    # create dataframe for each sequence
    df = pd.DataFrame(data[i], columns=['x','y'])

    for word in range(len(df)):
      if word == 0:
        # q(y1|START)
        dfA[df['y'][word]]['START'] += 1/len(data)
      elif word == len(df) - 1:
        # q(STOP|yn)
        #print("hi")
        dfA['STOP'][df['y'][word]] += 1/len(data)
      else:
        # q(yi|yi-1)
        total = labels[df['y'][word - 1]]
        dfA[df['y'][word]][df['y'][word - 1]] += 1/total

  stop = timeit.default_timer()
  print('Time: ', stop - start) 

  return dfA


# In[17]:


def viterbi_under(y, A, B, Pi=None):
    trans = A
    print(A.head(2))
    # drop STOP and START labels
    A = A.drop(labels='START', axis=0)
    A = A.drop("STOP",axis =1)
    A = A.to_numpy() # matrix for transition

    #convert text seq to num
    for i in range(len(y)):
      try:
        if(B.columns.get_loc(y[i])):
          y[i] = int(B.columns.get_loc(y[i]))
      
      except:
          y[i] = int(0) #set to unk

    B = B.to_numpy()

    K = A.shape[0]
    # Initialize the priors with default (uniform dist) if not given by caller
    Pi = Pi if Pi is not None else np.full(K, 1 / K)
    T = len(y)
    T1 = np.full((K, T), np.inf)
    T2 = np.full((K, T), np.inf)
    #print(T1)

    # Initialize first observation
    T1[:, 0] = np.log(trans.loc['START'][:-1].values) + np.log(B[:, int(y[0])])
    T2[:, 0] = 0
    #apply log to all
    #T1.apply(np.log)
    
    # Iterate throught the observations updating the tracking tables
    for i in range(1, T):
        T1[:, i] = np.max(T1[:, i - 1] + np.log(A.T) + np.log(B[np.newaxis, :, int(y[i])].T), 1)
        #T1[:, i] = np.max(np.log(T1[:, i - 1]) + np.log(A.T) + np.log(B[np.newaxis, :, y[i]].T, 1))
        #T2[:, i] = np.argmax(np.log(T1[:, i - 1]) + np.log(A.T)+ log(1))
        T2[:, i] = np.argmax(T1[:, i - 1] + np.log(A.T), 1)

    # Build the output, optimal model trajectory
    x = np.empty(T, 'B')
    x_out = np.empty(T,"object")
    x[-1] = np.argmax(T1[:, T - 1]) # get highest scoring last label
    for i in reversed(range(1, T)):
        x[i - 1] = T2[x[i], i]
    
    # map label index to index
    for i in range(len(x)):
      #print(x[i])
      
      x_out[i] = trans.columns[x[i]]
    print(x_out)
    return x_out,x, T1, T2


# In[18]:


def viterbi(y, A, B, Pi=None):
    """
    Return the MAP estimate of state trajectory of Hidden Markov Model.

    Parameters
    ----------
    y : array (T,)
        Observation state sequence. int dtype.
    A : array (K, K)
        State transition matrix. See HiddenMarkovModel.state_transition  for
        details.
    B : array (K, M)
        Emission matrix. See HiddenMarkovModel.emission for details.
    Pi: optional, (K,)
        Initial state probabilities: Pi[i] is the probability x[0] == i. If
        None, uniform initial distribution is assumed (Pi[:] == 1/K).

    Returns
    -------
    x : array (T,)
        Maximum a posteriori probability estimate of hidden state trajectory,
        conditioned on observation sequence y under the model parameters A, B,
        Pi.
    T1: array (K, T)
        the probability of the most likely path so far
    T2: array (K, T)
        the x_j-1 of the most likely path so far
    """
    trans = A
    print(A.head(2))
    A = A.drop(labels='START', axis=0)
    A = A.drop("STOP",axis =1)
    A = A.to_numpy() # matrix for transition

    #convert text seq to num
    for i in range(len(y)):
      try:
        if(B.columns.get_loc(y[i])):
          y[i] = int(B.columns.get_loc(y[i]))
      
      except:
          y[i] = int(0) #set to unk
    print(y[i])
    B = B.to_numpy()
    # Cardinality of the state space
    K = A.shape[0]
    # Initialize the priors with default (uniform dist) if not given by caller
    Pi = Pi if Pi is not None else np.full(K, 1 / K)
    T = len(y)
    T1 = np.empty((K, T), 'd')
    T2 = np.empty((K, T), 'B')
    #print(T1)

    # Initilaize the tracking tables from first observation
    T1[:, 0] = Pi * B[:, int(y[0])]
    T2[:, 0] = 0
    #apply log to all
    #T1.apply(np.log)
    
    # Iterate throught the observations updating the tracking tables
    for i in range(1, T):
        T1[:, i] = np.max(T1[:, i - 1] * A.T * B[np.newaxis, :, int(y[i])].T, 1)
        #T1[:, i] = np.max(np.log(T1[:, i - 1]) + np.log(A.T) + np.log(B[np.newaxis, :, y[i]].T, 1))
        #T2[:, i] = np.argmax(np.log(T1[:, i - 1]) + np.log(A.T)+ log(1))
        T2[:, i] = np.argmax(T1[:, i - 1] * A.T, 1)

    # Build the output, optimal model trajectory
    x = np.empty(T, 'B')
    x_out = np.empty(T,"object")
    x[-1] = np.argmax(T1[:, T - 1])
    for i in reversed(range(1, T)):
        x[i - 1] = T2[x[i], i]
    for i in range(len(x)):
      #print(x[i])
      
      x_out[i] = trans.columns[x[i]]
    print(x_out)
    return x_out,x, T1, T2


# In[19]:


def viterbi_kbest(y, A, B, Pi=None,k=7):
    """
    Return the MAP estimate of state trajectory of Hidden Markov Model.

    Parameters
    ----------
    y : array (T,)
        Observation state sequence. int dtype.
    A : array (K, K)
        State transition matrix. See HiddenMarkovModel.state_transition  for
        details.
    B : array (K, M)
        Emission matrix. See HiddenMarkovModel.emission for details.
    Pi: optional, (K,)
        Initial state probabilities: Pi[i] is the probability x[0] == i. If
        None, uniform initial distribution is assumed (Pi[:] == 1/K).

    Returns
    -------
    x : array (T,)
        Maximum a posteriori probability estimate of hidden state trajectory,
        conditioned on observation sequence y under the model parameters A, B,
        Pi.
    T1: array (K, T)
        the probability of the most likely path so far
    T2: array (K, T)
        the x_j-1 of the most likely path so far
    """
    trans = A
    #drop STOP and START labels
    A = A.drop("STOP",axis =1)
    A = A.drop("START",axis =0)
    Alen = A.shape[0]
    Alist = A.columns.tolist()
    A = A.to_numpy() # matrix for transition
    print(A.shape)

    #convert text seq to num
    for i in range(len(y)):
      try:
        if(B.columns.get_loc(y[i])):
          y[i] = int(B.columns.get_loc(y[i]))
      
      except:
          y[i] = int(0) #set to unk

    B = B.to_numpy()
    #print("Emis Shape: {}".format(B.shape))
    K = A.shape[0]
    # Initialize the priors with default (uniform dist) if not given by caller
    Pi = Pi if Pi is not None else np.full(K, 1 / K)
    T = len(y)  #length of input sequence
  
    #use np.zeros with 3d shape with 3rd dimension k=7
    T1= np.zeros((K,T,k),'d')
    T2 = np.zeros((K,T,k),'B')

    # Initialize first observation
    # START - label1
    temp = np.zeros((K,k), 'd')
    temp[:,k-1] = trans.loc['START'][:-1].values * B[:, int(y[0])] 
    T1[:,0,:] = temp

    # Forward
    # Iterate throught the observations updating the tracking tables
    for i in range(1, T):
      # for each column
      for j in range(0, K):
        # for each label
        # np.ravel: coordinates to int
        T1[j][i] = np.sort((T1[:, i - 1] * (A[:,j] * B[j][y[i]])[:, None]).ravel())[::-1][:k]
        
        #arg sort gives 2d shape, ravel to get scalar index do this
        T2[j][i] = np.argsort((T1[:, i - 1] * A[:,j][:,None]).ravel())[::-1][:k]

    # get sequence index of the k final highest scores 
    highest7 = np.argsort((T1[:, T-1] * trans['STOP'].values[1:][:, None]).ravel())[::-1][:k]
    print("highest: {}".format(highest7))
    
    # Build the output, optimal model trajectory
    x_out = []
    #x[-1] = np.arg7maxk(7,T1[:, T - 1]) # label of last word
    
    # Backtracking
    for tag in highest7:
      ls = [] # list of current label
      # np.unravel_index: int to coordinates
      ind = np.unravel_index(tag, (K,k))
      a,b = ind
      ls.append(Alist[a]) # label-STOP

      #for i in reversed(range(0, T)):
      for i in range(T-1,0,-1):
        if i != T-1:
          # if it is not the 2nd last word
          inde = np.unravel_index(ind, (K,k))
          a,b = inde
        
        #get parent index
        xprev = T2[a][i][b]
        # print("xprev: {}".format(xprev))
        ind = np.unravel_index(xprev, (K,k)) 
        c,d = ind
        # print("c: {}".format(c))
        # print("d: {}".format(d))
        
        ls.append(Alist[int(c)])
        ind = xprev
      x_out.append(ls[::-1])
      
    print("kbest sequence: {}".format(x_out[k-1]))

    return x_out[k-1]


# In[20]:


def viterbi_kunder(y, A, B, Pi=None,k=7):
    trans = A
    # drop START and STOP labels
    A = A.drop(labels='START', axis=0)
    A = A.drop("STOP",axis =1)
    Alist = A.columns.tolist() # list of all labels
    A = A.to_numpy() # matrix for transition

    #convert text seq to num
    for i in range(len(y)):
      try:
        if(B.columns.get_loc(y[i])):
          y[i] = int(B.columns.get_loc(y[i]))
      
      except:
          y[i] = int(0) #set to unk

    B = B.to_numpy()
    print("Emis Shape: {}".format(B.shape))
    K = A.shape[0]  #length of all labels
    # Initialize the priors with default (uniform dist) if not given by caller
    Pi = Pi if Pi is not None else np.full(K, 1 / K)
    T = len(y)  #length of input sequence
  
    #use np.zeros with 3d shape with 3rd dimension k=7
    T1= np.full((K,T,k), np.inf)
    T2 = np.full((K,T,k), np.inf)

    # Initilaize the tracking tables from first observation
    # START - label1
    temp = np.full((K,k), np.inf)
    temp[:,k-1] = np.log(trans.loc['START'][:-1].values) + np.log(B[:, int(y[0])]) 
    T1[:,0,:] = temp

    # Iterate throught the observations updating the tracking tables
    for i in range(1, T):
      # for each column
      for j in range(0, K):
        # for each tag
        # np.ravel: coordinates to int
        T1[j][i] = np.sort((T1[:, i - 1] + (np.log(A[:,j]) + np.log(B[j,int(y[i])]))[:, None]).ravel())[::-1][:k]
        #arg sort gives 2d shape, ravel to get scalar index do this
        T2[j][i] = np.argsort((T1[:, i - 1] + np.log(A[:,j])[:,None]).ravel())[::-1][:k]

    # get sequence index of the k final highest scores 
    highest7 = np.argsort((T1[:, T-1] + np.log(trans['STOP'].values[1:])[:, None]).ravel())[::-1][:k]
    print("highest: {}".format(highest7))
    
    # Build the output, optimal model trajectory
    x_out = []
    #x[-1] = np.arg7maxk(7,T1[:, T - 1]) # label of last word
    
    # Backtracking
    for tag in highest7:
      ls = [] # list of current label
      # np.unravel_index: int to coordinates
      ind = np.unravel_index(tag, (K,k))
      a,b = ind
      ls.append(Alist[a]) # label-STOP

      #for i in reversed(range(0, T)):
      for i in range(T-1,0,-1):
        if i != T-1:
          # if it is not the 2nd last word
          inde = np.unravel_index(int(ind), (K,k))
          a,b = inde
        
        #get parent index
        xprev = T2[a][i][b]
        # print("xprev: {}".format(xprev))
        ind = np.unravel_index(int(xprev), (K,k)) 
        c,d = ind
        # print("c: {}".format(c))
        # print("d: {}".format(d))
        
        ls.append(Alist[int(c)])
        ind = xprev
      x_out.append(ls[::-1])
      

    return x_out[k-1]


# In[22]:


import copy
def train_perceptron(EN_train,trans,emis,pi,epoch,lr):
    trans = trans
    emis = emis
    scores =[]
    best_emis = trans
    best_trans =  emis
    currentHigh = 100000
    
    for k in range(epoch):
        print(k)
        word_len = 0
        count = 0
        for i in range(len(EN_train)):
            word_n_label = EN_train[i]
            sequence = []
            label = []
            for j in range(len(word_n_label)):
                sequence.append(EN_train[i][j][0])
                label.append(EN_train[i][j][1])
            words = np.copy(sequence)
            word_len += len(words)
            x_out,output,t1,t2 = viterbi(sequence,trans,emis,pi)
        #x_out gives me the labels
            for j in range(len(sequence)):
                if label[j] != x_out[j]:#this means that predicted label is wrong, we need to adjust the matrices
                #print("wrong_label :{}".format(x_out[j]))
                #print(x_out[j])
                #print(words[j])
                    if (emis.loc[x_out[j],words[j]] >0):
                        emis.loc[x_out[j],words[j]] -= 1*lr
                    if (emis.loc[label[j],words[j]]< 1):
                        emis.loc[label[j],words[j]]+= 1*lr
                    if trans.loc[label[j-1],label[j]] <1: 
                        trans.loc[label[j-1],label[j]] += 1*lr
                    if(trans.loc[x_out[j-1],x_out[j]])>0: 
                        trans.loc[x_out[j-1],x_out[j]]-= 1*lr
                    count +=1
        scores.append([count,k])
        print(count)
        
        if count <currentHigh:
            print("Updated Matrix!")
            best_trans = trans
            best_emis = emis
            currentHigh =  count
                
        
    return best_trans,best_emis,scores

                        
            
        
        


# In[ ]:





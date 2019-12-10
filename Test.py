#!/usr/bin/env python
# coding: utf-8

# In[23]:


# test set
from Functions import *
EN_tpath = 'EN/'
SG_tpath = '/content/drive/My Drive/ML Data/SG/'
AL_tpath = '/content/drive/My Drive/ML Data/AL/'
CN_tpath = '/content/drive/My Drive/ML Data/CN/'


def part2(path):
    train = load_file(path)
    x, labelcount, trans, emis = init_values(train)
    emis = emission(emis,train,labelcount)
    tag  = sentiment(emis)
    return tag


def part2_test(path):
  tag = part2(path+"train")
  data = []

  # output file
  out = open(path + 'dev.p2.out', 'w')

  # load test data
  with open(path + 'dev.in') as fp:
      for empty, line in groupby(fp, lambda x: x.startswith('\n')):
          if not empty:
              data.append(np.array([[str(x).rstrip() for x in d.split()] for d in line if len(d.strip())]))

  print(data[0])
  for sequence in data:
    for word in sequence:
      if word[0] not in tag:
        label = tag["#UNK#"]    # replace word if do not appear in modified training set
      else:
        label = tag[word[0]]
    
      string = word[0] + " " + label
      out.write("%s\n" % string)

    out.write("\n") # new sequence

    
    
    


# In[ ]:


part2_test(EN_tpath)
part2_test(SG_tpath)
part2_test(AL_tpath)
part2_test(CN_tpath)


# In[16]:


def part3(path):
    train = load_file(path)
    x, labelcount, trans, emis = init_values(train)
    emis = emission(emis,train,labelcount)
    #tag  = sentiment(emis)
    trans = transition(trans,train,labelcount)
    return trans,emis
def part3_test(path,lang):
  data = []

  # output file
  out = open(path + 'dev.p3u.out', 'w')

  # load test data
  with open(path + 'dev.in') as fp:
      #for empty, line in groupby(fp, lambda x: x.startswith('\n')):
          #if not empty:
              #data.append(np.array([[str(x).rstrip() for x in d.split()] for d in line if len(d.strip())]))
      lines = []
      for line in fp:
        if line != "\n":
          lines.append(line.strip("\n"))
        else:
          data.append(lines)
          lines = []
  a,b = part3(path+"train")

  # y_list = list(df) # list of all labels aka df[y]
  # a = df.to_numpy() # matrix for transition
  pi = np.full(a.shape[0], 1/a.shape[0]) # init pi

  for sequence in data:
      temp = list.copy(sequence)
      #print("this seq is : {}".format(sequence))
      #print("temp is : {}".format(temp))

      x_out,output,t1,t2 = viterbi_under(temp,a,b,pi)
      # write to output file
      #for i in range(len(sequence)):
       # print(sequence[i])
       #string =  str(temp[i]) + " " + str(x_out[i])
       # out.write("%s\n" % string)

     #   out.write("\n") # new sequence
      count = 0
      for i in sequence:
        out.write("{} {}\n".format(i,x_out[count]))
        count += 1
      out.write("\n")
  #print("Written p3u.out: {}".format(lang))


    


# In[18]:


part3_test(EN_tpath)
part3_test(SG_tpath)
part3_test(AL_tpath)
part3_test(CN_tpath)


# In[22]:


def part4_test(path):
  data = []

  # output file
  out = open(path + 'dev.p4u.out', 'w')

  # load test data
  with open(path + 'dev.in') as fp:
      #for empty, line in groupby(fp, lambda x: x.startswith('\n')):
          #if not empty:
              #data.append(np.array([[str(x).rstrip() for x in d.split()] for d in line if len(d.strip())]))
      lines = []
      for line in fp:
        if line != "\n":
          lines.append(line.strip("\n"))
        else:
          data.append(lines)
          lines = []

  

  #y_list = list(df) # list of all labels aka df[y]
  #a = df.to_numpy() # matrix for transition
  #pi = np.full(a.shape[0], 1/a.shape[0]) # init pi
  a,b = part3(path+"train")

  for sequence in data:
      temp = list.copy(sequence)
      #print("this seq is : {}".format(sequence))
      #print("temp is : {}".format(temp))

      x_out = viterbi_kunder(temp,a,b)

      # write to output file
      count = 0
      for i in sequence:
        out.write("{} {}\n".format(i,x_out[count]))
        count += 1
      out.write("\n")
  #print("Written p4u.out: {}".format(lang))


# In[20]:


part4_test(EN_tpath)
part4_test(SG_tpath)
part4_test(AL_tpath)
part4_test(CN_tpath)


# In[31]:


def part5(path):
    train = load_file(path)
    x, labelcount, trans, emis = init_values(train)
    emis = emission(emis,train,labelcount)
    #tag  = sentiment(emis)
    trans = transition(trans,train,labelcount)
    
    return trans,emis,train
def part5_test(path,epoch,lr):
    a,b,train = part5(path+"train")
    df = (a.drop(labels='START', axis=0))
    df = df.drop("STOP",axis =1)
    pi = np.full(a.shape[0], 1/a.shape[0]) # init pi
    a,b,score,useless = train_perceptron(train,df,b,pi,epoch,lr)
    data = []

  # output file
    out = open(path + 'dev.p5.out', 'w')

  # load test data
    with open(path + 'test.in') as fp:
      #for empty, line in groupby(fp, lambda x: x.startswith('\n')):
          #if not empty:
              #data.append(np.array([[str(x).rstrip() for x in d.split()] for d in line if len(d.strip())]))
        lines = []
        for line in fp:
            if line != "\n":
                lines.append(line.strip("\n"))
            else:
                data.append(lines)
                lines = []

    for sequence in data:
        temp = list.copy(sequence)
      #print("this seq is : {}".format(sequence))
      #print("temp is : {}".format(temp))

        x_out = viterbi_kunder(temp,a,b)

      # write to output file
        count = 0
        for i in sequence:
            out.write("{} {}\n".format(i,x_out[count]))
            count += 1
        out.write("\n")
    


# In[32]:


part5_test(EN_tpath,10,0.0005)
#part5_test(SG_tpath,10,0.0005)
part5_test(AL_tpath,10,0.0005)
#part5_test(CN_tpath,10,0.0005)


# In[ ]:





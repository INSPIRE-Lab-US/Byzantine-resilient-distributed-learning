import numpy as np

def dimensionwise_median(g, b):
   M = len(g)
   if b >= M/2:
       raise ValueError("Byzantine nodes must be less than half of total nodes")
   screened_grad = []
   for layer in range(len(g[0])):
       sort = [node[layer] for node in g]
       sort = np.sort(sort, axis = 0)
       screened_grad.append(sort[int(M/2)])
   
   return screened_grad

def dimensionwise_trimmed_mean(g, b):
   M = len(g)
   if b >= M/2:
       raise ValueError("Byzantine nodes must be less than half of total nodes")
   screened_grad = []
   for layer in range(len(g[0])):
       sort = [node[layer] for node in g]
       sort = np.sort(sort, axis = 0)
       screened_grad.append(np.mean(sort[b : -b], axis = 0))
   return screened_grad
def geometric_median(g, b, m=1):
   pass

def flatten(l):
   out = []
   def flat_vec(layer): 
       for i in layer: 
           if type(i) == list or type(i) is np.ndarray: 
               flat_vec(i) 
           else: 
               out.append(i)
   flat_vec(l) 
   return out

def krum(g, b, m=1):
   if m >= len(g) - b - 2:
       raise ValueError("m is too large")
   #score is calculated only by the last layer since the whole vector is too large
   g = list(g)
   g_vec = [g_nd[-2] for g_nd in g]
   output=[]
   for r in range(m):
       #score is the distance to the closest m-b-2 gradients
       score = []
       for _g in g_vec:
           dist = [np.linalg.norm(np.array(_g) - np.array(other)) for other in g_vec]
           dist.sort()
           score.append(np.sum(dist[: (len(g) - b - 2)]))

       ind = score.index(min(score))
       output.append(g[ind])
       del g_vec[ind]
       del g[ind]
   output = np.mean(output, axis=0)
   return output
       
def bulyan(g, b):
   M = len(g)
   if M < 4 * b + 3:
       raise ValueError("M has to be large than 4b+2")
   #score is calculated only by the last layer since the whole vector is too large
   g = list(g)
   g_vec = [g_nd[-2] for g_nd in g]
   A=[]
   for r in range(M-2*b):
       #score is the distance to other gradients
       score = []
       for _g in g_vec:
           dist = [np.linalg.norm(np.array(_g) - np.array(other)) for other in g_vec]
           dist.sort()
           score.append(np.sum(dist[: (len(g) - b - 2)]))
       ind = score.index(min(score))
       A.append(g[ind])
       del g_vec[ind]
       del g[ind]
   screened = dimensionwise_trimmed_mean(A, b)       
   return screened

def geo_med(g, b):
   pass

def zeno(g, b, oracle):
   M = len(g)
   if b >= M:
       raise ValueError("M has to be large than b")
   #score is calculated only by the last layer since the whole vector is too large
   g = list(g)
   g_vec = [np.array(flatten(g_nd[-2])) for g_nd in g]
   ref = np.array(flatten(oracle[-2]))
   output = []
   n_ref = np.linalg.norm(ref)
   #score increases when close to oracle and decreases when norm is large
   score = []
   for _g in g_vec:
       n_g = np.linalg.norm(_g)
       c = n_ref / n_g
       _g = _g * c
       score.append(np.inner(_g, ref) - n_g * 10)  #This weight should be tuned      
   for r in range(M - b):
       ind = score.index(max(score))
       output.append(g[ind])
       del score[ind]
       del g[ind]
   output = np.mean(output, axis=0)
   return output

def sign_sgd(g, b):
   M = len(g)
   if b >= M/2:
       raise ValueError("Byzantine nodes must be less than half of total nodes")
   screened_grad = []
   for layer in range(len(g[0])):
       signed = [np.sign(node[layer]) for node in g]
       screened_grad.append(np.mean(signed, axis = 0))    
   return screened_grad    

   
Byzantine_algs ={'dimensionwise_median': dimensionwise_median, 
                'dimensionwise_trimmed_mean': dimensionwise_trimmed_mean,
               'krum': krum,
               'bulyan': bulyan,
               'zeno': zeno,
               'sign_sgd': sign_sgd} 

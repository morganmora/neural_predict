import numpy as np
import neural

class Key(object):
    """
        represetn the key itself made from 5 number
        integers from 1 to 50
    """
    def __init__(self, key = []):
        
        self.key = key


    def get_fingerprint(self):
        
        accum = 0
        
        for member in self.key:
            accum = 2 ** member
        
        return accum


    def get_binary_representation(self):
        
        bin_array=[]
        
        for i in range(0, 51):
            bin_array.append(0)
            
            if i in self.key:
                bin_array[i] = 1
        
        return bin_array



def print_representation(key):
    k = Key(key)

    print (key)
    print(k.get_binary_representation())
    print(k.get_fingerprint())
    return



"""
like create electronic black box , lot of unknown component, with input and output
find value in the blackbox, the refine with more componets, kind of reverse-engineer.

"""

#3, 12, 36, 44, 45 
#8, 16, 24, 26, 35
#7, 17, 29, 37, 45
#2, 4, 8, 27, 50
#6, 15, 20, 30, 38
#5, 7, 21, 25, 37
#3, 8, 10, 32, 45
#1, 3, 33, 40, 45
#9 18 32 38 46

#X = (hours studying, hours sleeping), y = score on test, xPredicted = 4 hours studying & 8 hours sleeping (input data for prediction)
#X = np.array(([3, 8, 10, 32, 45], [5, 7, 21, 25, 37], [6, 15, 20, 30, 38], [2, 4, 8, 27, 50],[8, 16, 24, 26, 35]), dtype=int)
#y = np.array(( [5, 7, 21, 25, 37], [6, 15, 20, 30, 38], [2, 4, 8, 27, 50], [7, 17, 29, 37, 45],[8, 16, 24, 26, 35]), dtype=int)
#xPredicted = np.array(([7, 17, 29, 37, 45]), dtype=int)

X = np.array(([3, 8, 10, 32, 45],[6, 15, 20, 30, 38],[7, 17, 29, 37, 45],[9, 18, 32, 38, 46]), dtype=int)
y = np.array(([5, 7, 21, 25, 37], [2, 4, 8, 27, 50], [8, 16, 24, 26, 35],[1, 3, 33, 40, 45]), dtype=int)
xPredicted = np.array(([3, 12, 36, 44, 45]), dtype=int)

# scale units
X = X/50 # maximum of X array
xPredicted = xPredicted/50 # maximum of xPredicted (our input data for the prediction)
y = y/50 # max test score is 100


NN = neural.Network(5,5,25)
for i in range(10000): # trains the NN 1,000 times
  #print ("# " + str(i) + "\n")
  #print ("Input (scaled): \n" + str(X))
  #print ("Actual Output: \n" + str(y))
  #print ("Predicted Output: \n" + str(NN.forward(X)))
  #print ("Loss: \n" + str(np.mean(np.square(y - NN.forward(X))))) # mean sum squared loss
  #print ("\n")
  NN.train(X, y)


NN.saveWeights()
NN.predict(xPredicted)


def get_fingerprint(key = []):
        
        accum = 0
        
        for member in key:
            accum = 2 ** member
        
        return accum


def get_binary_representation(key = []):
        
        bin_array=[]
        
        for i in range(0, 51):
            bin_array.append(0)
            
            if i in key:
                bin_array[i] = 1
        
        return bin_array


###Test number 2 with binary arrays
X = np.array((get_binary_representation([3, 8, 10, 32, 45]),
get_binary_representation([6, 15, 20, 30, 38]),
get_binary_representation([7, 17, 29, 37, 45])), dtype=int)

y = np.array((get_binary_representation([5, 7, 21, 25, 37]), 
get_binary_representation([2, 4, 8, 27, 50]),
get_binary_representation([8, 16, 24, 26, 35])), dtype=int)

xPredicted = np.array(get_binary_representation([3, 12, 36, 44, 45]), dtype=int)


N2 = neural.Network(51,51,2601)
for i in range(10000): # trains the NN 1,000 times
  #print ("# " + str(i) + "\n")
  #print ("Input (scaled): \n" + str(X))
  #print ("Actual Output: \n" + str(y))
  #print ("Predicted Output: \n" + str(NN.forward(X)))
  #print ("Loss: \n" + str(np.mean(np.square(y - N2.forward(X))))) # mean sum squared loss
  #print ("\n")
  N2.train(X, y)


N2.saveWeights()
N2.predict(xPredicted)





#行列Wを固定として、与えられたグラフGに対してhGを求めるGNN関数またはクラスを実装
import numpy as np
class GNN(self,D,W,T):
    def __init__(self,D,W,T):
        self.D = D
        self.W = W
        self.T = T

    def relu(self,x):
        return np.maximum(0,x)
    
    def aggregate1(self,G,x):
        a = np.dot(G,x)
        return a
    
    def aggregate2(self , a):
        x = self.frompyfunc()
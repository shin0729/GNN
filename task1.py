#行列Wを固定として、与えられたグラフGに対してhGを求めるGNN関数またはクラスを実装
import numpy as np
class GNN:
    def __init__(self,D,W,T):
        self.D = D
        self.W = W
        self.T = T
    @staticmethod
    def relu(x):
        return np.maximum(0,x)
    
    def f(self,x):
        return np.frompyfunc(self.relu,1,1)(x)

    def aggregate(self,G,x):
        a = np.dot(G,x)
        return a
    
    def conbine(self , a):
        h_v = self.f(np.dot(a,self.W))
        return h_v
    
    def readout(self,G):
        h_G = np.zeros((G.shape[0],self.D))
        h_G = np.random.rand(G.shape[0],self.D)
        for _ in range(self.T):
            a = self.aggregate(G,h_G)
            h_G = self.conbine(a)
        return h_G

#main関数
if __name__ == "__main__":
    G = np.array([[0,0,1,0,0],
                  [0,0,1,0,0],
                  [1,1,0,1,1],
                  [0,0,1,0,1],
                  [0,0,1,1,0]])
    D = 3
    W = np.random.rand(D,D)
    print(W)
    T = 2
    gnn = GNN(D,W,T)
    h_G = gnn.readout(G)
    print(h_G)

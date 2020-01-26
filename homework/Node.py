import numpy as np
class plus_node():
    def __init__(self):
        self.x,self.y,self.z = None,None,None
    def forward(self,x,y):
        self.x,self.y,self.z = x,y,x+y
        return self.z
    def backward(self,dL):
        return dL, dL
class plus5_node():
    def __init__(self):
        self.x1,self.x2,self.x3,self.x4,self.x5,self.z = None, None, None, None, None, None
    def forward(self,x1,x2,x3,x4,x5):
        self.x1,self.x2,self.x3,self.x4,self.x5,self.z = x1,x2,x3,x4,x5,x1+x2+x3+x4+x5
        return self.z
    def backward(self,dL):
        return dL,dL,dL,dL,dL
class plus4_node():
    def __init__(self):
        self.x1,self.x2,self.x3,self.x4,self.z = None, None, None, None, None
    def forward(self,x1,x2,x3,x4):
        self.x1,self.x2,self.x3,self.x4,self.z = x1,x2,x3,x4,x1+x2+x3+x4
        return self.z
    def backward(self,dL):
        return dL,dL,dL,dL
class minus_node():
    def __init__(self):
        self.x,self.y,self.z = None,None,None
    def forward(self,x,y):
        self.x, self.y, self.z = x, y, x - y
        return self.z
    def backward(self,dL):
        return dL, -1 * dL
##### Your Code(Forward Propagation) #####
class mul_node():
    def __init__(self):
        self.x, self.y, self.z = None, None, None
        
    def forward(self, x, y):
        self.x, self.y, self.z = x, y, x*y
        return self.z
    def backward(self, dL):
        return self.y*dL, self.x*dL   
class square_node():
    def __init__(self):
        self.x, self.z = None, None
    
    def forward(self, x):
        self.x, self.z = x, x*x
        return self.z
    
    def backward(self, dL):
        return 2*self.x*dL
class cost_node():
    def __init__(self):
        self.x, self.z = None, None
    
    def forward(self, x):
        self.x = x
        self.z = np.mean(self.x)
        return self.z
    def backward(self):
        return 1/len(self.x)*np.ones(shape = (len(self.x)))
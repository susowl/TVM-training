from typing import OrderedDict
import tvm
from tvm import topi
import math
import numpy as np
import os

os.environ['TVM_NUM_THREADS'] = '16'

class SGD:
    """
        Classical SGD with momentum
    """
    def __init__(self,net,loss,LR=0.0001,momentum_factor=0.0,) -> None:
        """
        Create SGD object

        Args:
            net: Network to optimize
            loss: Loss function
            LR (float, optional): Learning rate. Defaults to 0.0001.
            momentum_factor (float, optional): Momentum factor. Defaults to 0.
        """
        self.new_params = []
        self.params = []
        self.momentum = []
        self.new_momentum = []
        self._LR = tvm.nd.array(np.float32(LR))
        self._momentum_factor= tvm.nd.array(np.float32(momentum_factor))
        self.net = net
        self.loss = loss
        self.forward_backward= None

    @property
    def LR(self):
        """
        Set LR

        Returns:
            float: current LR value
        """
        return float(self._LR.numpy())
    
    @LR.setter
    def LR(self,newLR):
        self._LR = tvm.nd.array(np.float32(newLR))

    @property
    def momentum_factor(self):
        """
        Set momentum factor

        Returns:
            float: current momentum value
        """
        return float(self._LR.numpy())
    
    @momentum_factor.setter
    def momentum_factor(self,new_momentum):
        self._momentum_factor = tvm.nd.array(np.float32(new_momentum))

    
    def createGraph(self,shape):
        """
        Creates training pipeline graph:
            input_shape
             |
            net  #createGraph
             |
            loss #createGraph
             |
            grads
             |
            momentum = momentum factor*momentum + (1.0-momentum_factor)*grads
            network_params = network_params - LR*momentum

        Graph inputs:
            x - imput batch
            labels - labels for loss
            params - current network params
            momentums - list of cutrrent momentum values
            momentum_factor,LR - SGD params
        Graph outputs:
            y - loss value
            net_out - raw network output before loss
            new_momentums - list of new momentum values
            new_params - list of new network params

        Method called by SGD.compile method.

        Args:
            shape (_type_): _description_

        Returns:
            _type_: _description_
        """
        inp = tvm.te.placeholder(shape, name='x',dtype = "float32")
        labels = tvm.te.placeholder((inp.shape[0],), name='labels',dtype = "int")

        x = self.net.createGraph(inp)
        net_out = x
        params = list(self.net.params_placeholder.values())
        y = self.loss.createGraph(x,labels)

        #calclate weight gradients
        grad = tvm.te.gradient(y, params)

        momentum_factor = tvm.te.placeholder((), name='momentum_factor',dtype = "float32")
        LR = tvm.te.placeholder((), name='LR',dtype = "float32")
        momentums = [tvm.te.placeholder(p.shape, name=p.name+"_momentum",dtype = "float32") for p in params]

        new_momentum = [momentum_factor*m +(1.0-momentum_factor)*g for m,g in zip(momentums,grad)]
        newW = [w-LR*m for w,m in zip(params,new_momentum)]
        
        return y,net_out,inp,labels,params,momentum_factor,LR,momentums,list(new_momentum),newW

    def compile(self,shape,target="llvm"):
        """
        Creates a training graph and build it.
        Args:
            shape (tupple): int tuple with imput shape
            target (str, optional): compiler target and options. Defaults to "llvm".
        """
        y,net_out,inp,labels,params,momentum_factor,LR,momentums,new_momentum,newW = self.createGraph(shape)
        sched = tvm.te.create_schedule([y.op,net_out.op]+[nW.op for nW in newW]+[nM.op for nM in new_momentum])
        self.forward_backward = tvm.build(sched, [y,net_out,inp,labels]+params+[momentum_factor,LR]+momentums+new_momentum+newW,target=target)

        # create some buffers for input params,momentums and output params and momentums
        for placeholder in self.net.params_placeholder:
            shape = [int(v) for v in self.net.params_placeholder[placeholder].shape] #convert intimm to normal int!
            if placeholder.endswith("weight"):
                #Apply kaiming xavier init to weights                
                limit = np.sqrt(1/(math.prod(shape[1:])))

                self.params.append((tvm.nd.array(np.random.uniform(-limit,limit,size=shape).astype("float32"))))
                self.new_params.append(tvm.nd.array(np.random.uniform(size=shape).astype("float32")))
                self.momentum.append(tvm.nd.array(np.zeros(shape=shape).astype("float32")))
                self.new_momentum.append(tvm.nd.array(np.zeros(shape=shape).astype("float32")))
            else:
                self.params.append(tvm.nd.array(np.zeros(shape=shape).astype("float32")))
                self.new_params.append(tvm.nd.array(np.zeros(shape=shape).astype("float32")))
                self.momentum.append(tvm.nd.array(np.zeros(shape=shape).astype("float32")))
                self.new_momentum.append(tvm.nd.array(np.zeros(shape=shape).astype("float32")))
    
    def step(self,inp,out,labels):
        """
        Apply SGD step

        Args:
            inp (tvm.nd.array): current batch [batch size x C x W x H]
            out (tvm.nd.array): buffer for raw network output [batch size x class count}
            labels (tvm.nd.array): labels for loss [batch size]

        Returns:
            (y,out): y: loss value ; out: raw network output
        """
                
        y = tvm.nd.array(np.float32(0))
        ret = self.forward_backward(*([y,out,inp,labels]+self.params+[self._momentum_factor,self._LR]+self.momentum+self.new_momentum+self.new_params))
        self.params,self.new_params,self.momentum,self.new_momentum = self.new_params,self.params,self.new_momentum,self.momentum
        return y,out

class CrossEntropyLoss:
    """
        Simple softmax cross-entropy loss
    """
    def __init__(self) -> None:
        self.params_placeholder = {}

    def createGraph(self,inp,labels):
        """
        Creates softmax cross-entropy loss graph:
        inp           labels
         |               |
        logsoftmax       |
         |               |
        nll_loss---------
         |
        sum
         |
        *1/batch_size

        Args:
            inp : network output [batch size x class count]
            labels : label values [batch size]

        Returns:
            y: loss value
        """
        x = topi.nn.log_softmax(inp)
        y = topi.nn.nll_loss(x,labels,topi.full_like(labels, 1.0),'ave',ignore_index=-1)
        y = topi.sum(y)/inp.shape[0]
        return y

class LeNet:
    """
    Classical 28 x 28 Lenet-5 for MNIST
    """
    def __init__(self) -> None:
        self.params_placeholder = OrderedDict()  # list of all traininble params

    def conv(self,inp,filter_count:int,kernel_size=3,stride = 1,pad = 0,dilation=1,bias=True,name=None):
        """
        Adds 2D convolution layer to network.

        Args:
            inp: input tensor
            filter_count (int): Filter count
            kernel_size (int, optional): Kernel size. Defaults to 3.
            stride (int, optional): kernel stride. Defaults to 1.
            pad (int, optional): padding. Defaults to 0.
            dilation (int, optional): kernel dialtion. Defaults to 1.
            bias (bool, optional): apply bias. Defaults to True.
            name (_type_, optional):unique layer name. If None generates automaticaly. Defaults to None.

        Returns:
            out: result tensor
        """
        name = 'conv_{}'.format(len(self.params_placeholder)) if name == None else name
        w = tvm.te.placeholder((filter_count, inp.shape[1], kernel_size, kernel_size), name='{}_weight'.format(name),dtype = "float32")
        self.params_placeholder[name+"_weight"] = w
        out = topi.nn.conv2d(inp,w,stride,pad,dilation)
        if bias:
            b = tvm.te.placeholder((1, filter_count, 1, 1), name='{}_bias'.format(name),dtype = "float32")
            out = out + b
            self.params_placeholder[name+"_bias"]=b
        return out

    def dense(self,inp,filter_count,bias=True,name = None):
        """
        Adds dense layer to network. Also automaticaly convert input tensor shape.

        Args:
            inp : input tensor
            filter_count : filter count
            bias (bool, optional): apply bias. Defaults to True.
            name (_type_, optional):unique layer name. If None generates automaticaly. Defaults to None.

        Returns:
            out: result tensor
        """
        name = 'fc_{}'.format(len(self.params_placeholder)) if name == None else name
        if len(inp.shape)>2:
            inp = topi.reshape(inp,(inp.shape[0],math.prod(inp.shape[1:]))) #reshape (n,-1) not supported
        w = tvm.te.placeholder((filter_count, inp.shape[1],), name='dense_{}_weight'.format(name),dtype = "float32")
        self.params_placeholder[name+"_weight"] = w
        b = None
        if bias:
            b = tvm.te.placeholder((filter_count, ), name='dense_{}_bias'.format(name),dtype = "float32")
            self.params_placeholder[name+"_bias"]=b

        return topi.nn.dense(inp,w,b)
    
    def createGraph(self,inp):
        """
        Create LeNet-5 network graph.

        Args:
            inp : input tensot

        Returns:
            out: result tensor
        """
        x = self.conv(inp,32,5,pad=2,bias=True)
        x = topi.nn.relu(x)
        x = topi.nn.pool2d(x,[2,2],[2,2],[1,1],[0,0,0,0],'max')
        x = self.conv(x,64,5,bias=True)
        x = topi.nn.relu(x)
        x = topi.nn.pool2d(x,[2,2],[2,2],[1,1],[0,0,0,0],'max')
        x = self.dense(x,120)
        x = topi.nn.relu(x)
        x = self.dense(x,84)
        x = topi.nn.relu(x)
        x = self.dense(x,10)
        x = topi.nn.relu(x)

        return x

def main():
    batchSize = 32
    #create network
    net = LeNet()
    #create loss function
    loss = CrossEntropyLoss()
    #create SGD solver and attach network and loss
    solver = SGD(net,loss,LR=0.1)

    #compile training graph using input shape 32x1x28x28
    #solver.compile((32,1,28,28),"llvm -opt-level=0")   #for debug - builds very fast
    solver.compile((batchSize,1,28,28),"llvm -mcpu=core-avx2")  

    import time
    curT = time.time()

    from mnistDB import mnistDB
    DB = mnistDB()

    # Apply one epoch
    for batch_idx, (data, target) in enumerate(DB.trainDataloader(batchSize)):
        #Normalize input
        data = (data/255.0 - 0.1307)/0.3081
        data = data.reshape((batchSize,1,28,28)).astype("float32")
        input_tvm = tvm.nd.array(data)
        lbls = tvm.nd.array(target.astype("int32"))
        out = tvm.nd.array(np.zeros((32,10)).astype("float32"))
        res = solver.step(input_tvm,out,lbls)
        answers = res[1].numpy().argmax(1)
        acc = (answers==target).sum()/data.shape[0]
        curSpeed = (time.time()-curT)/(batch_idx+1)
        print("TVM:{} | accuracy:{} | second per iteration:{}".format(res[0],acc,curSpeed))

if __name__ == '__main__':
    main()
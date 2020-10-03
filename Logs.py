import cupy as cp 
import numpy as np
from IPython.display import clear_output
import matplotlib.pyplot as plt


class Logs():
  def __init__(self,types):
    self.logs = []
    self.types = []
    self.counter = 0
    self.output_dir = []
    for t,it in zip(types,np.arange(len(types))):
      print(t)
      self.types.append(t)
      self.logs.append([])
      self.output_dir.append("pictures/" + t[1] + "/")
      self.mkdir_p(self.output_dir[it])
  
  def find_division(self,x):
    res = [1,x]
    for i in range(2,int(np.floor(np.sqrt(x))+1)):
      if x % i == 0:
        res = [int(i),int(x/i)]
    res = [int(np.ceil(x/3)),3]
    return res

  def log(self,values):
    for value,typee,it in zip(values,self.types,np.arange(len(self.types))):
      if typee[0] == 'number':
        self.logs[it].append(value)
      if typee[0] == 'matrix' or typee[0] == 'population':
        if typee[0] == 'matrix':
          value = cp.asnumpy(value) - np.diag(np.ones(value.shape[0]))
        self.logs[it] = cp.asnumpy(value)
      if typee[0] == 'vector':
        self.logs[it] = cp.asnumpy(value)

  def mkdir_p(self,mypath):
    '''Creates a directory. equivalent to using mkdir -p on the command line'''

    from errno import EEXIST
    from os import makedirs,path

    try:
        makedirs(mypath)
    except OSError as exc: # Python >2.5
        if exc.errno == EEXIST and path.isdir(mypath):
            pass
        else: raise

  def plot(self):
    clear_output()
    #sizes = self.find_division(len(self.types))
    #fig, axes = plt.subplots(sizes[0],sizes[1], figsize = (24,14)) #maybe calculate figsize?
    for log,it in zip(self.types,np.arange(len(self.types))):
      #idx = (int(np.floor(it/sizes[1])),int(it%sizes[1]))
      fig = plt.figure(figsize = (24,20))
      ax = fig.add_subplot(111)
      if log[0] == 'number':
        ax.plot(np.arange(len(self.logs[it])),self.logs[it])
        #axes[idx[0]][idx[1]].plot(np.arange(len(self.logs[it])),self.logs[it])
        #axes[idx[0]][idx[1]].grid(True)
        #axes[idx[0]][idx[1]].set_title(log[1] + ':')
      if log[0] == 'matrix':
        cax = ax.matshow(self.logs[it])
        fig.colorbar(cax)
        #cax = axes[idx[0]][idx[1]].matshow(self.logs[it])
        #fig.colorbar(cax)
        #axes[idx[0]][idx[1]].grid(True)
        #axes[idx[0]][idx[1]].set_title(log[1] + ':')
      if log[0] == 'population':
        ax.scatter(np.arange(self.logs[it].shape[1]),np.mean(self.logs[it],axis=0))
        #axes[idx[0]][idx[1]].scatter(np.arange(self.logs[it].shape[1]),np.mean(self.logs[it],axis=0))
        #axes[idx[0]][idx[1]].grid(True)
        #axes[idx[0]][idx[1]].set_title(log[1] + ':')
      if log[0] == 'vector':
        ax.scatter(np.arange(self.logs[it].shape[0]),self.logs[it])
        #axes[idx[0]][idx[1]].scatter(np.arange(self.logs[it].shape[0]),self.logs[it])
        #axes[idx[0]][idx[1]].grid(True)
        #axes[idx[0]][idx[1]].set_title(log[1] + ':')
      ax.set_title(log[1] + ':',fontsize = 15)
      fig.savefig(self.output_dir[it] + log[1] + str(self.counter))
      plt.show()
    self.counter += 1
    #fig.tight_layout()
    #plt.show()
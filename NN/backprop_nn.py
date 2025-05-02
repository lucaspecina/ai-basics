# ----------------------------------------------------------------------
# Backprop for running in debug mode
# ----------------------------------------------------------------------

import math
import random
import numpy as np
import matplotlib.pyplot as plt


from graphviz import Digraph

def trace(root):
  # builds a set of all nodes and edges in a graph
  nodes, edges = set(), set()
  def build(v):
    if v not in nodes:
      nodes.add(v)
      for child in v._prev:
        edges.add((child, v))
        build(child)
  build(root)
  return nodes, edges

def draw_dot(root):
  dot = Digraph(format='svg', graph_attr={'rankdir': 'LR'}) # LR = left to right
  
  nodes, edges = trace(root)
  for n in nodes:
    uid = str(id(n))
    # for any value in the graph, create a rectangular ('record') node for it
    dot.node(name = uid, label = "{ %s | data %.4f | grad %.4f }" % (n.label, n.data, n.grad), shape='record')
    if n._op:
      # if this value is a result of some operation, create an op node for it
      dot.node(name = uid + n._op, label = n._op)
      # and connect this node to it
      dot.edge(uid + n._op, uid)

  for n1, n2 in edges:
    # connect n1 to the op node of n2
    dot.edge(str(id(n1)), str(id(n2)) + n2._op)

  dot.attr(size='15')
  return dot


class Value:
  
  def __init__(self, data, _children=(), _op='', label=''):
    self.data = data
    self.grad = 0.0
    self._backward = lambda: None
    self._prev = set(_children)
    self._op = _op
    self.label = label

  def __repr__(self):
    return f"Value(data={self.data})"
  
  # operations
  
  def __add__(self, other):
    other = other if isinstance(other, Value) else Value(other)
    out = Value(self.data + other.data, (self, other), '+')
    
    def _backward():
      self.grad += 1.0 * out.grad
      other.grad += 1.0 * out.grad
    out._backward = _backward
    
    return out

  def __mul__(self, other):
    other = other if isinstance(other, Value) else Value(other)
    out = Value(self.data * other.data, (self, other), '*')
    
    def _backward():
      self.grad += other.data * out.grad
      other.grad += self.data * out.grad
    out._backward = _backward
      
    return out
  
  def __pow__(self, other):
    assert isinstance(other, (int, float)), "only supporting int/float powers for now"
    out = Value(self.data**other, (self,), f'**{other}')

    def _backward():
        self.grad += other * (self.data ** (other - 1)) * out.grad
    out._backward = _backward

    return out
  
  def __rmul__(self, other): # other * self
    return self * other

  def __truediv__(self, other): # self / other
    return self * other**-1

  def __neg__(self): # -self
    return self * -1

  def __sub__(self, other): # self - other
    return self + (-other)

  def __radd__(self, other): # other + self
    return self + other

  def tanh(self):
    x = self.data
    t = (math.exp(2*x) - 1)/(math.exp(2*x) + 1)
    out = Value(t, (self, ), 'tanh')
    
    def _backward():
      self.grad += (1 - t**2) * out.grad
    out._backward = _backward
    
    return out
  
  def exp(self):
    x = self.data
    out = Value(math.exp(x), (self, ), 'exp')
    
    def _backward():
      self.grad += out.data * out.grad # NOTE: in the video I incorrectly used = instead of +=. Fixed here.
    out._backward = _backward
    
    return out
  
  
  def backward(self):
    
    topo = []
    visited = set()
    def build_topo(v):
      if v not in visited:
        visited.add(v)
        for child in v._prev:
          build_topo(child)
        topo.append(v)
    build_topo(self)
    
    self.grad = 1.0
    for node in reversed(topo):
      node._backward()



#########################

# __call__ sirve para cuando queres llamar directamente al objeto

class Neuron:
  
  def __init__(self, nin, id=''): # nin: number of inputs (x)
    self.w = [Value(random.uniform(-1,1), label=f'{id}_W{i}') for i in range(nin)]
    self.b = Value(random.uniform(-1,1), label=f'{id}_b')
  
  def __call__(self, x):
    # a(w * x + b)
    act = sum((wi*xi for wi, xi in zip(self.w, x)), self.b)
    out = act.tanh()
    return out
  
  def parameters(self):
    return self.w + [self.b]


#########################

class Layer:
  
  def __init__(self, nin, nout, id=''): # n inputs, n outputs (cantidad de neuronas)
    self.nin = nin
    self.neurons = [Neuron(nin, f"L{id}_n{i}") for i in range(nout)]
  
  def __call__(self, x):
    outs = [n(x) for n in self.neurons] # ejecuta cada neurona con la data
    return outs[0] if len(outs) == 1 else outs # devuelve los resultados individuales
  
  def parameters(self):
    # para cada neurona te devuelve sus parameters
    return [p for neuron in self.neurons for p in neuron.parameters()]
  

#########################

class MLP:
  
  def __init__(self, nin, nouts): 
    """n inputs (x), 
    nouts: lista donde la cantidad de elementos es una layer y su value es cantidad de neuronas por layer. El final se cuenta como una layer mas
    """
    sz = [nin] + nouts
    self.layers = [Layer(sz[i], sz[i+1], str(i)) for i in range(len(nouts))]
  
  def __call__(self, x):
    for layer in self.layers:
      x = layer(x)
    return x
  
  def parameters(self):
    return [p for layer in self.layers for p in layer.parameters()]
  

#########################

def main():
    # TRAINING DATA
    xs = [
    [2.0, 3.0, -1.0],
    [3.0, -1.0, 0.5],
    [0.5, 1.0, 1.0],
    [1.0, 1.0, -1.0],
    ]
    ys = [1.0, -1.0, -1.0, 1.0] # desired targets

    # INITIALIZATION
    n = MLP(3, [4, 4, 1])
    layers = n.layers
    for l in layers:
        print(f'{l}: inputs: {l.nin}, neurons: {len(l.neurons)}')

    iterations = 100
    losses = []

    for k in range(iterations):
        # forward pass
        ypred = [n(x) for x in xs]
        loss = sum((yout - ygt)**2 for ygt, yout in zip(ys, ypred))
        
        # backward pass
        for p in n.parameters():
            p.grad = 0.0
        loss.backward()
        
        # update
        for p in n.parameters():
            p.data += -0.1 * p.grad
        
        losses.append(loss.data)
        if iterations <= 20:
            print(k, loss.data)

    for ygt, yout in zip(ys, ypred):
        print(f'target: {ygt}, pred: {yout.data}, loss ({yout.data} - {ygt})**2 = {((yout - ygt)**2).data}')


if __name__ == "__main__":
  main()


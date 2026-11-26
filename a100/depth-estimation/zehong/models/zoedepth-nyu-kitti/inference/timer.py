"""仅对实际网络调用记录设备Event；不包输入搬运、翻转或输出后处理。"""
class Timer:
 def __init__(self,api):self.api,self.enabled,self.events,self.originals=api,False,[],[]
 def call(self,fn,*args,**kwargs):
  if not self.enabled:return fn(*args,**kwargs)
  start,end=self.api.Event(enable_timing=True),self.api.Event(enable_timing=True)
  start.record();out=fn(*args,**kwargs);end.record();self.events.append((start,end));return out
 def wrap(self,obj,name):
  original=getattr(obj,name)
  def wrapped(*args,**kwargs):return self.call(original,*args,**kwargs)
  setattr(obj,name,wrapped);self.originals.append((obj,name,original))
 def milliseconds(self):
  if not self.events:raise RuntimeError('No device network events were recorded')
  self.api.synchronize();return sum(a.elapsed_time(b) for a,b in self.events)
 def close(self):
  for obj,name,fn in self.originals:setattr(obj,name,fn)

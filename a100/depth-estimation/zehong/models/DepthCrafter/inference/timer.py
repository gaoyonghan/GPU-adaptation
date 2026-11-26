import torch,functools
class NetworkTimer:
 def __init__(self,device):self.api=getattr(torch,device.type);self.events=[];self.restore=[]
 def wrap(self,obj,method):
  original=getattr(obj,method)
  @functools.wraps(original)
  def timed(*args,**kwargs):
   begin,end=self.api.Event(enable_timing=True),self.api.Event(enable_timing=True)
   begin.record();result=original(*args,**kwargs);end.record();self.events.append((begin,end));return result
  setattr(obj,method,timed);self.restore.append((obj,method,original))
 def reset(self):self.events=[]
 def seconds(self):
  if not self.events:raise RuntimeError('No network calls recorded')
  self.events[-1][1].synchronize();return sum(a.elapsed_time(b) for a,b in self.events)/1000
 def close(self):
  for obj,name,fn in reversed(self.restore):setattr(obj,name,fn)

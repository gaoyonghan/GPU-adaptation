from transformers import SegformerForImageClassification
def load_model(weights):
 model,info=SegformerForImageClassification.from_pretrained(str(weights),local_files_only=True,output_loading_info=True)
 if any(info.get(k) for k in ['missing_keys','unexpected_keys','mismatched_keys','error_msgs']):raise RuntimeError('Checkpoint/model mismatch: '+str(info))
 # Original batch evaluator uses timm transform (not the single-image demo processor).
 return model,None

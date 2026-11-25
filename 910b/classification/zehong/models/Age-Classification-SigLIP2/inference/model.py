from transformers import SiglipForImageClassification,AutoImageProcessor
def load_model(weights):
 model,info=SiglipForImageClassification.from_pretrained(str(weights),local_files_only=True,output_loading_info=True)
 if any(info.get(k) for k in ['missing_keys','unexpected_keys','mismatched_keys','error_msgs']):raise RuntimeError('Checkpoint/model mismatch: '+str(info))
 return model.float(),AutoImageProcessor.from_pretrained(str(weights),local_files_only=True,use_fast=False)

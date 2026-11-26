from inference.native.modeling_mambavision import MambaVisionModelForImageClassification
def load_model(weights):
 model,info=MambaVisionModelForImageClassification.from_pretrained(str(weights),local_files_only=True,output_loading_info=True)
 if any(info.get(k) for k in ['missing_keys','unexpected_keys','mismatched_keys','error_msgs']):raise RuntimeError('Checkpoint/model mismatch: '+str(info))
 if model.config.num_labels!=1000:raise ValueError('Expected verified 1000-class checkpoint')
 return model,None

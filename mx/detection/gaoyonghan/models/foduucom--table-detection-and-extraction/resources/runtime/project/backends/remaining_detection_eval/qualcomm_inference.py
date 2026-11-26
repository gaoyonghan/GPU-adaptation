"""CPU ONNX pipeline using frozen Qualcomm v0.61.0 geometry functions."""
from pathlib import Path
import cv2
import numpy as np
import onnxruntime as ort
import torch
import qualcomm_geometry as geo

ROOT=Path(__file__).resolve().parent/'assets/qualcomm'

class Pipeline(geo.NativeROI):
    def __init__(self,kind):
        self.kind=kind
        root=ROOT/f'mediapipe_{kind}-onnx-float'
        options=ort.SessionOptions();options.intra_op_num_threads=2;options.inter_op_num_threads=1
        self.det=ort.InferenceSession(str(next(root.rglob(f'{kind}_detector.onnx'))),sess_options=options,providers=['CPUExecutionProvider'])
        self.land=ort.InferenceSession(str(next(root.rglob(f'{kind}_landmark_detector.onnx'))),sess_options=options,providers=['CPUExecutionProvider'])
        self.anchors=torch.tensor(np.load(ROOT/'source'/('anchors_palm.npy' if kind=='hand' else 'anchors_face_back.npy')),dtype=torch.float32).reshape(-1,2,2)
        self.keypoint_rotation_vec_start_idx=0 if kind=='hand' else 1
        self.keypoint_rotation_vec_end_idx=2 if kind=='hand' else 0
        self.rotation_offset_rads=np.pi/2 if kind=='hand' else 0
        self.detect_box_offset_xy=.5 if kind=='hand' else 0
        self.detect_box_scale=2.5 if kind=='hand' else 1.1
        self.size=256 if kind=='hand' else 192

    def predict(self,image):
        array=np.asarray(image).copy();tensor=geo.numpy_image_to_torch(array)
        inputs,scale,padding=geo.resize_pad(tensor,(256,256))
        outputs=self.det.run(None,{'image':inputs.numpy()})
        if self.kind=='hand':coords,scores=outputs
        else:coords=np.concatenate(outputs[:2],axis=1);scores=np.concatenate(outputs[2:],axis=1)
        coords,scores=geo.mediapipe_detector_postprocess(torch.from_numpy(coords),torch.from_numpy(scores),100,(256,256),self.anchors)
        selected,conf=geo.batched_nms(.3,.001,coords,scores)
        detections=[]
        for raw,score in zip(selected[0][:300],conf[0][:300]):
            raw=raw.reshape(-1,2).clone();geo.denormalize_coordinates(raw,(1,1),scale,padding)
            box=raw[:2].reshape(-1).tolist()
            if box[2]<=box[0] or box[3]<=box[1]:continue
            pred={'bbox_xyxy':box,'score':float(score),'detector_keypoints':raw[2:].tolist()}
            # Sparse detection AP uses .001; expensive landmark stage uses frozen .5.
            if score>=.5:
                vector=raw[2+self.keypoint_rotation_vec_start_idx]-raw[2+self.keypoint_rotation_vec_end_idx]
                if torch.linalg.vector_norm(vector)==0:
                    pred['landmark_failure']='coincident_rotation_keypoints_after_native_pixel_quantization'
                    detections.append(pred)
                    continue
                roi=self._compute_object_roi([raw[:2][None]],[raw[2:][None]])[0]
                assert torch.isfinite(roi).all(), 'Nonfinite ROI'
                affine=geo.compute_box_affine_crop_resize_matrix(roi[:,:3],(self.size,self.size))[0]
                crop=cv2.warpAffine(array,affine,(self.size,self.size))
                result=self.land.run(None,{'image':geo.numpy_image_to_torch(crop).numpy()})
                land_score=float(result[0][0]);landmarks=result[-1][0].copy()
                assert landmarks.shape==((21,3) if self.kind=='hand' else (468,3)) and np.isfinite(landmarks).all()
                inverse=cv2.invertAffineTransform(affine)
                xy=landmarks[:,:2]*self.size
                xy=np.c_[xy,np.ones(len(xy))]@inverse.T
                z=landmarks[:,2]*self.size*np.linalg.norm(inverse[:,0])
                assert np.isfinite(xy).all() and np.isfinite(z).all(), 'Nonfinite restored landmarks'
                pred.update(landmark_score=land_score,landmarks=np.c_[xy,z].tolist())
                if self.kind=='hand':pred['right_hand_probability']=float(result[1][0])
            detections.append(pred)
        return detections

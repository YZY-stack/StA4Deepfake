import numpy as np
import cv2
import random
import yaml
import pickle
from tqdm import tqdm
from PIL import Image as pil_image
import dlib
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional as F
import torch.utils.data
from torchvision import transforms
from trainer.trainer import Trainer
from detectors import DETECTOR
from collections import defaultdict
from PIL import Image as pil_image
from imutils import face_utils
from skimage import transform as trans
import os
from os.path import join


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def inference(model, data_dict):
    data, label = data_dict['image'], data_dict['label']
    # move data to GPU
    data_dict['image'], data_dict['label'] = data.to(device), label.to(device)
    predictions = model(data_dict, inference=True)
    return predictions


# preprocess the input image --> cropped face, resize = 256, adding a dimension of batch (output shape: 1x3x256x256)
def get_keypts(image, face, predictor, face_detector):
    # detect the facial landmarks for the selected face
    shape = predictor(image, face)
    
    # select the key points for the eyes, nose, and mouth
    leye = np.array([shape.part(37).x, shape.part(37).y]).reshape(-1, 2)
    reye = np.array([shape.part(44).x, shape.part(44).y]).reshape(-1, 2)
    nose = np.array([shape.part(30).x, shape.part(30).y]).reshape(-1, 2)
    lmouth = np.array([shape.part(49).x, shape.part(49).y]).reshape(-1, 2)
    rmouth = np.array([shape.part(55).x, shape.part(55).y]).reshape(-1, 2)
    
    pts = np.concatenate([leye, reye, nose, lmouth, rmouth], axis=0)

    return pts

def extract_aligned_face_dlib(face_detector, predictor, image, res=256, mask=None):
    def img_align_crop(img, landmark=None, outsize=None, scale=1.3, mask=None):
        """ 
        align and crop the face according to the given bbox and landmarks
        landmark: 5 key points
        """

        M = None
        target_size = [112, 112]
        dst = np.array([
            [30.2946, 51.6963],
            [65.5318, 51.5014],
            [48.0252, 71.7366],
            [33.5493, 92.3655],
            [62.7299, 92.2041]], dtype=np.float32)

        if target_size[1] == 112:
            dst[:, 0] += 8.0

        dst[:, 0] = dst[:, 0] * outsize[0] / target_size[0]
        dst[:, 1] = dst[:, 1] * outsize[1] / target_size[1]

        target_size = outsize

        margin_rate = scale - 1
        x_margin = target_size[0] * margin_rate / 2.
        y_margin = target_size[1] * margin_rate / 2.

        # move
        dst[:, 0] += x_margin
        dst[:, 1] += y_margin

        # resize
        dst[:, 0] *= target_size[0] / (target_size[0] + 2 * x_margin)
        dst[:, 1] *= target_size[1] / (target_size[1] + 2 * y_margin)

        src = landmark.astype(np.float32)

        # use skimage tranformation
        tform = trans.SimilarityTransform()
        tform.estimate(src, dst)
        M = tform.params[0:2, :]

        # M: use opencv
        # M = cv2.getAffineTransform(src[[0,1,2],:],dst[[0,1,2],:])

        img = cv2.warpAffine(img, M, (target_size[1], target_size[0]))

        if outsize is not None:
            img = cv2.resize(img, (outsize[1], outsize[0]))
        
        if mask is not None:
            mask = cv2.warpAffine(mask, M, (target_size[1], target_size[0]))
            mask = cv2.resize(mask, (outsize[1], outsize[0]))
            return img, mask
        else:
            return img

    # Image size
    height, width = image.shape[:2]

    # Convert to rgb
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Detect with dlib
    faces = face_detector(rgb, 1)
    if len(faces):
        # For now only take the biggest face
        face = max(faces, key=lambda rect: rect.width() * rect.height())
        
        # Get the landmarks/parts for the face in box d only with the five key points
        landmarks = get_keypts(rgb, face, predictor, face_detector)

        # Align and crop the face
        cropped_face = img_align_crop(rgb, landmarks, outsize=(res, res), mask=mask)
        cropped_face = cv2.cvtColor(cropped_face, cv2.COLOR_RGB2BGR)
        
        # Extract the all landmarks from the aligned face
        face_align = face_detector(cropped_face, 1)
        landmark = predictor(cropped_face, face_align[0])
        landmark = face_utils.shape_to_np(landmark)

        return cropped_face, landmark,face
    
    else:
        return None, None



def prepare_model(detector_path, weights_path):
    #load the detector config
    with open(detector_path, 'r') as f:
        config = yaml.safe_load(f)

    #load the pretrained weights
    config['weights_path'] = weights_path
    
    # prepare the model (detector)
    model_class = DETECTOR[config['model_name']]
    model = model_class(config, demo=True).to(device)
    
    # load the pretrained weights
    ckpt = torch.load(weights_path, map_location=device)
    filtered_ckpt = {k.replace('module.', ''): v for k, v in ckpt.items()}
    # import pdb;pdb.set_trace()
    model.load_state_dict(filtered_ckpt, strict=True)
    print('===> Load checkpoint done!')


    return model



def single_video_inference_for_VidModel(video_path, face_detector, face_predictor,  model, output_path,
                            start_frame=0, end_frame=None, cuda=False):
    model.eval()
    print('Starting: {}'.format(video_path))

    if cuda:
        model = model.cuda()
    
    # Read and write
    reader = cv2.VideoCapture(video_path)

    video_fn = video_path.split('/')[-1].split('.')[0]+'.avi'
    os.makedirs(output_path, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    fps = reader.get(cv2.CAP_PROP_FPS)
    num_frames = int(reader.get(cv2.CAP_PROP_FRAME_COUNT))
    writer = None

    print(f'This video contains {num_frames} frames in total. Using all of them for inference.')

    # Frame numbers and length of output video
    frame_num = 0
    assert start_frame < num_frames - 1
    end_frame = end_frame if end_frame else num_frames
    pbar = tqdm(total=end_frame-start_frame)

    img_list = []
    print('=> starting extracting frames from the given video')
    while reader.isOpened():
        _, image = reader.read()
        if image is None:
            break
        frame_num += 1

        if frame_num < start_frame:
            continue
        pbar.update(1)

        # Image size
        height, width = image.shape[:2]

        # Init output writer
        if writer is None:
            writer = cv2.VideoWriter(join(output_path, video_fn), fourcc, fps,
                                     (height, width)[::-1])


        # FIXME: res is fixed to be 224 for my video model
        cropped_face, landmarks,face = extract_aligned_face_dlib(face_detector, face_predictor, image, res=224, mask=None)
        #224,224,3 --> 1,3,224,224

        preprocess = transforms.Compose([
            transforms.ToTensor(),
            # FIXME: mean and std should be these values
            transforms.Normalize(mean = [0.48145466, 0.4578275, 0.40821073], std = [0.26862954, 0.26130258, 0.27577711])
        ])

        cropped_face = cv2.cvtColor(cropped_face, cv2.COLOR_BGR2RGB)
        cropped_face = preprocess(pil_image.fromarray(cropped_face))

        cropped_face = cropped_face.unsqueeze(0)
        img_list.append(cropped_face)
        
        if frame_num >= end_frame:
            break

        # Show
        # cv2.imshow('test', image)
        # cv2.waitKey(33)     # About 30 fps
        # writer.write(image)
    pbar.close()


    clip_list = collect_clip_for_one_video(img_list)



    # ====== video inference ======
    print('=> starting inference.')
    print(f'=> The input video contains {len(clip_list)} clips')
    prob_list = []
    for i, one_clip in enumerate(clip_list):
        data_dict = {}
        data_dict['image'] = torch.cat(one_clip).unsqueeze(0)
        data_dict['label'] = torch.from_numpy(np.array([0]))

        predictions = inference(model, data_dict)  # contain: logits, prob, feat
        prob = round(predictions['prob'].detach().cpu().numpy()[0], 4)
        print(f"clip_{i+1}/{len(clip_list)}: fake prob is {prob}")
        prob_list.append(prob)


    print(f'Finished! The overal fake prob of this video is {np.mean(prob_list)}')


def collect_clip_for_one_video(frame_paths=None, clip_size=8):
    total_frames = len(frame_paths)
    assert clip_size==8, "my model requires the clip_size to be 8"
    if total_frames < clip_size:
        raise ValueError(f"The video is too short, containing less than {clip_size} effective frames to be recognized and cropped")

    # Initialize an empty list to store the selected continuous frames
    selected_clips = []

    # Calculate the number of clips to select
    num_clips = total_frames // clip_size

    # Calculate the step size between each clip
    clip_step = (total_frames - clip_size) // (num_clips - 1)

    # Select clip_size continuous frames from each part of the video
    for i in range(num_clips):
        # Ensure start_frame + clip_size - 1 does not exceed the index of the last frame
        start_frame = random.randrange(i * clip_step, min((i + 1) * clip_step, total_frames - clip_size + 1))
        continuous_frames = frame_paths[start_frame:start_frame + clip_size]
        assert len(continuous_frames) == clip_size, 'clip_size is not equal to the length of frame_path_list'
        selected_clips.append(continuous_frames)

    return selected_clips



def main():
    #path to detector(Xception), weights, image, face_detector, face-predictor
    detector_path = './training/config/detector/clip_adapter.yaml'
    weights_path = './training/weights/sta_best.pth'
    face_detector = dlib.get_frontal_face_detector()
    predictor_path = './preprocessing/dlib_tools/shape_predictor_81_face_landmarks.dat'
    face_predictor = dlib.shape_predictor(predictor_path)
    video_path = './simswap_000_003.mp4'
    output_path = './'

    model = prepare_model(detector_path, weights_path)

    #————————————————single image inference————————————————
    # print('Starting: {}'.format(image_path))
    # # Read the image
    # image = cv2.imread(image_path)
    

    # cls_prob, label, image = single_image_inference(image, face_detector, face_predictor,model)

    # # Visualize
    # cv2.imshow('test', image)
    # key = cv2.waitKey(10000)

    # print('Model prediction:\nlabel:', label, "\nprobability(deepfake):", cls_prob[1])
    # print('===> Test Done!')


    #————————————————single video inference—————————————————

    single_video_inference_for_VidModel(video_path, face_detector, face_predictor,  model, output_path,
                            start_frame=0, end_frame=None, cuda=True)




if __name__ == '__main__':
    main()

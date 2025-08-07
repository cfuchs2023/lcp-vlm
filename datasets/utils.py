# -*- coding: utf-8 -*-
import torch
import os
from abc import ABC, abstractmethod
import cv2
import clip
import numpy as np    

#%%
prompts_templates = {
"UCF101":'f\'a photo of a person doing {c}.\'', 
"hmdb51":'f\'a photo of a person doing {c}.\'', 
"Kinetics400":'f\'a photo of a person doing {c}.\'',
'sun397':'f\'a photo of a {c}.\'',
'Food101':'f\'a photo of a {c}, a type of food.\'',
'dtd':'f\'a photo of a person doing {c}.\'',
'OxfordPets':'f\'a photo of a {c}, a type of pet.\'',
'eurosat':'f\'a centered satellite photo of {c}.\'',
'StanfordCars':'f\'a photo of a {c}, a type of car.\'',
'Caltech101':'f\'a photo of a {c}.\'',
'Flower102':'f\'a photo of a {c}, a type of flower.\'',
'fgvc_aircraft':'f\'a photo of a {c}, a type of aircraft.\'',
'imagenet':'f\'a photo of a {c}.\'',
#'UCF101':FeaturesLoader

}
#%%
class AbstractDataloader(ABC):
    def __init__(self, root_path, dataset_name):
        self.dataset_name = dataset_name
        self.root_path = os.path.join(root_path, dataset_name)
        self.get_labels_names()
        
    
    def __extract_frames(self, video_output_folder, video_path, num_frames = 10, ):
        """Extracts a specified number of frames from a video."""
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_indices = [
            int(i * total_frames / num_frames) for i in range(num_frames)
        ]  # Uniformly distribute frames
        
        count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if count in frame_indices:
                frame_filename = os.path.join(video_output_folder, f"frame_{count:04d}.jpeg")
                cv2.imwrite(frame_filename, frame)
            count += 1
        cap.release()
        return None
    
    def get_textual_prototypes(self, clip_model, prompts = None, model_dim = 512, return_all = False):
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        if prompts is None:
            prompts = self.get_text_prompts()

        # Encode prompts
        tokenized_prompts = {}
        all_prompts = {}
        K = len(prompts.keys())
        d = model_dim

        avg_prompts = torch.zeros((K,d), dtype = torch.float16)
        for j,cname_ in enumerate(prompts.keys()):
            cname = cname_#.replace('_', ' ')
            #print(f'class : {cname}, num_prompts : {len(cupl_prompts[cname])}')
            tokenized_prompts[cname] = clip.tokenize(prompts[cname]).to(device)
            with torch.autocast("cuda"), torch.no_grad():
                encoded_p_cname = clip_model.encode_text(tokenized_prompts[cname].to(device))
            all_prompts[cname] = encoded_p_cname / torch.linalg.norm(encoded_p_cname, dim = -1, keepdims = True)
            encoded_p_cname = encoded_p_cname/torch.linalg.norm(encoded_p_cname, dim = -1, keepdims = True)
            avg_prompts[j,...] = encoded_p_cname.mean(0)
        
        clip_prototypes = (avg_prompts/torch.linalg.norm(avg_prompts, dim = -1, keepdims = True)).T.to(torch.float32).cpu()

        if return_all:
            return clip_prototypes, all_prompts
        else:
            return clip_prototypes

    def get_text_prompts(self, label_names = None, template = None  ):
        if template is None: 
            template = prompts_templates[self.dataset_name]
        if label_names is None:
            self.get_labels_names()
            label_names = list(self.labels_names)
        prompts = {}
        #print('len label names in get_text_prompts : ', len(label_names))
        for c in label_names:
            if c in prompts.keys():
                #print(c)
                if c == 'missile':
                    c = 'missiles'
                elif c == 'sunglasses':
                    c = "sunglass"
                else:
                    raise RuntimeError('Good luck')
            prompts[c] = eval(template)# f'A photo of {c}, a human action.' #f'a photo of a person doing {c}.'
        
        return prompts 
    
    def get_ensemble_text_prompts(self, label_names = None):
        if label_names is None:
            self.get_labels_names()
            label_names = list(self.labels_to_idx.keys())
        templates = [ #'a photo of a person doing %s.',
                      'someone performing %s.',
                      'a movie frame of someone doing %s.',
                      #'a person doing %s.',
                      'a movie character doing %s.',
                      'a frame from a youtube video of someone doing %s.',
                      #'a photo of a person practicing %s.',
                      'a photo of someone fighting %s.',
                      #'a photo of a person fighting %s.'
                      ]
        prompts = {}
        for c in label_names:
            prompts[c]  = []
            for temp in templates:
                prompts[c].append(temp % c)
        return prompts, templates
    
    @abstractmethod
    def get_labels_names(self, *args, **kwargs):
        pass
    
    @abstractmethod
    def ExtractFrames(self, *args, **kwargs):
        pass
    
    @abstractmethod
    def EncodeFrames(self, *args, **kwargs):
        pass
    
    @abstractmethod
    def load_features_and_labels(self, *args, **kwargs):
        pass
#%%    
def get_soft_labels(temperature, features, text_embeddings):
    logits = temperature*features.type(torch.float32) @ text_embeddings
    soft_labels = logits.softmax(dim=-1)
    return soft_labels

def calib_test_sets(
        features:torch.Tensor|np.ndarray, 
        labels:torch.Tensor|np.ndarray, 
        n_shots:int=50, 
        seed:int=0
    ) -> tuple[torch.Tensor|np.ndarray, torch.Tensor|np.ndarray, torch.Tensor|np.ndarray, torch.Tensor|np.ndarray]:
    """
    Splits the features set and label set into a calibration and a test set.
    
    Parameters:
        features (torch.Tensor or np.ndarray): Feature array of shape (n_samples, n_features).
        labels (torch.Tensor or np.ndarray): Array of labels of shape (n_samples,).
        n_shots (int): Number of samples per class in the calibration set.
        seed (int): Random seed for reproducibility.

    Returns:
        calib_features (torch.Tensor or np.ndarray): The features of the points that are part of the calibration dataset.
        calib_labels (torch.Tensor or np.ndarray): The labels of the points that are part of the calibration dataset.
        test_features (torch.Tensor or np.ndarray): The features of the points that are part of the test dataset.
        test_labels (torch.Tensor or np.ndarray): The labels of the points that are part of the test dataset.
    """
    rng = np.random.RandomState(seed)
    all_labels = np.unique(labels)
    selected_indices = np.zeros(len(all_labels)*n_shots, dtype=int)

    for l, label in enumerate(all_labels):
        # Get indices of the current integer
        indices = np.where(labels == label)[0]
        
        # Randomly select the required number of indices
        selected_indices[l*n_shots: (l+1)*n_shots] = rng.choice(indices, n_shots, replace=False)

    mask = np.zeros_like(labels, dtype=bool)
    mask[selected_indices] = 1

    calib_features, calib_labels = features[mask], labels[mask]
    test_features, test_labels = features[~mask], labels[~mask]
    calib_shuffle = rng.permutation(len(calib_labels))
    test_shuffle = rng.permutation(len(test_labels))

    return calib_features[calib_shuffle], calib_labels[calib_shuffle], test_features[test_shuffle], test_labels[test_shuffle]

def calib_test_sets_folds(
        features:torch.Tensor|np.ndarray, 
        labels:torch.Tensor|np.ndarray, 
        total_n_samples = 1000,
        n_folds:int=1,
        normalize:bool=True,
        seed:int=0
    ) -> tuple[list, list, list, list]:
    """
    Splits features and labels into multiple calibration and test sets using n_folds.

    Parameters:
        features (torch.Tensor or np.ndarray): Feature array of shape (n_samples, n_features).
        labels (torch.Tensor or np.ndarray): Array of labels of shape (n_samples,).
        n_shots (int): Number of samples per class in each calibing set.
        n_folds (int): Number of folds to generate.
        normalize (bool): Whether the output features should be normalized or not.
        seed (int): Random seed for reproducibility.

    Returns:
        calib_features_list (list) List of calib features for each fold.
        calib_labels_list (list) List of calib labels for each fold.
        test_features_list (list) List of test features for each fold.
        test_labels_list (list) List of test labels for each fold.
    """
    if total_n_samples is not None:
        print('Using total_n_samples for sampling')
    
    rng = np.random.RandomState(seed)
    all_labels = np.unique(labels)
    
    calib_features_list, calib_labels_list = [], []
    test_features_list, test_labels_list = [], []
    
    for _fold in range(n_folds):
        
        selected_indices = rng.choice(len(labels), total_n_samples, replace=False)
            
        # Create mask for selected indices
        mask = np.zeros_like(labels, dtype=bool)
        mask[selected_indices] = 1
        
        # Split into calib and test sets
        calib_features, calib_labels = features[mask], labels[mask]
        test_features, test_labels = features[~mask], labels[~mask]

        if normalize:
            calib_features = _normalize_features(calib_features)
            test_features = _normalize_features(test_features)

        # Shuffle calib and test sets
        calib_shuffle = rng.permutation(len(calib_labels))
        test_shuffle = rng.permutation(len(test_labels))
        
        calib_features_list.append(calib_features[calib_shuffle])
        calib_labels_list.append(calib_labels[calib_shuffle])
        test_features_list.append(test_features[test_shuffle])
        test_labels_list.append(test_labels[test_shuffle])
            
    
    return calib_features_list, calib_labels_list, test_features_list, test_labels_list

def _normalize_features(features:torch.Tensor) -> torch.Tensor:
    norm_features  = features.mean(1)
    norm_features /= torch.linalg.norm(norm_features , dim = -1, keepdims = True)
    return norm_features

#%%
import json
import pickle
import torch
from scipy.io import loadmat
import pandas as pd
dataset_names_folder_to_zhou = {#'imagenet':'',
                                'sun397':'SUN397',
                                'Food101':'Food101',
                                'dtd':'DescribableTextures',
                                'OxfordPets':'OxfordPets',
                                'eurosat':'EuroSAT',
                                'StanfordCars':'StanfordCars',
                                'Caltech101':'Caltech101',
                                'Flower102':'OxfordFlowers',
                                'UCF101':'UCF101',
                                'fgvc_aircraft':'fgvc_aircraft',
                                'imagenet':'imagenet'}

class FeaturesLoader(AbstractDataloader):       
        
    def get_labels_names(self):
        if self.dataset_name in  dataset_names_folder_to_zhou.keys():
            if self.dataset_name not in ['Flower102', 'StanfordCars', 'fgvc_aircraft', 'imagenet']: #, 
                zhou_name = dataset_names_folder_to_zhou[self.dataset_name]
                with open(os.path.join(self.root_path, f'split_zhou_{zhou_name}.json'), 'r') as f:
                    zhou_split = json.load(f)
                li_labels = []
                for u in zhou_split['test']:
                    if u[-1] not in li_labels:
                        li_labels.append(u[-1])
                #li_labels = sorted(li_labels)
                self.labels_names = li_labels
                self.labels_to_idx = {}
                self.idx_to_labels = {}
                for jlabel, label in enumerate(li_labels):
                    self.labels_to_idx[label] = jlabel
                    self.idx_to_labels[jlabel] = label
            elif self.dataset_name == 'Flower102':
                with open(os.path.join(self.root_path, 'cat_to_name.json'), 'r') as f:
                    cat_to_names = json.load(f)
                self.idx_to_labels = cat_to_names
                self.labels_to_idx = {}
                li_labels = []
                li_idx = [int(u) for u in self.idx_to_labels.keys()]
                for i in range(len(li_idx)):
                    idx_ = str(i+1)
                    li_labels.append(self.idx_to_labels[idx_])
                    self.labels_to_idx[self.idx_to_labels[idx_]] = i
                self.labels_names = li_labels
            elif self.dataset_name == 'StanfordCars':
                label_file = pd.read_csv(os.path.join(self.root_path, 'archive', 'names.csv'),
                                        header=None, names=['raw'])
                li_labels = []
                for index, row in label_file.iterrows():
                    label = row.iloc[0]
                    #print(index,row[0])
                    li_labels.append(label)
                    #print(li_labels)
                self.labels_names = li_labels
                self.labels_to_idx = {}
                self.idx_to_labels = {}
                for jlabel, label in enumerate(li_labels):
                    self.labels_to_idx[label] = jlabel
                    self.idx_to_labels[jlabel] = label
            elif self.dataset_name == 'fgvc_aircraft':
                li_labels = []
                with open(os.path.join(self.root_path, 'variants.txt'), 'r') as f:
                    lines = f.readlines()
                    for line in lines:
                        li_labels.append(line.strip())
                self.labels_names = li_labels
                self.labels_to_idx = {}
                self.idx_to_labels = {}
                for jlabel, label in enumerate(li_labels):
                    self.labels_to_idx[label] = jlabel
                    self.idx_to_labels[jlabel] = label
            elif self.dataset_name == 'imagenet':
                from .imagenet import imagenet_classes
                #print('num imagenet classes : ', len(imagenet_classes))
                li_labels = [u.replace('_', ' ') for u in imagenet_classes]
                self.labels_names = li_labels
                
                self.labels_to_idx = {}
                self.idx_to_labels = {}
                for jlabel, label in enumerate(li_labels):
                    self.labels_to_idx[label] = jlabel
                    self.idx_to_labels[jlabel] = label

                
        return None
    
    def load_features_and_labels(self, backbone_name, split_data = False,  splits = ['test', 'val', 'train']):
        bname = backbone_name.replace('/', '_').replace('-','_')
        if self.dataset_name == 'imagenet':
            print('Loading ImageNet. Replacing splits with [\'test\']')
            splits = ['test']

        out = []
        all_features = []
        all_labels = []
        for split in splits:
            path = os.path.join(self.root_path , 'cache',f'{bname}_{split}_features.pt')
            features = torch.load(path, weights_only=True)
            all_features.append(features)
            
            path = os.path.join(self.root_path , 'cache',f'{split}_target.pt')
            labels = torch.load(path, weights_only=True)
            all_labels.append(labels)
            
        if split_data:
            for j in range(len(all_labels)):
                out.append(all_features[j])
                out.append(all_labels[j])
        else:
            all_features = torch.cat(all_features, dim = 0)
            all_labels = torch.cat(all_labels)
            out = [all_features, all_labels] + [None for _ in range(2*(len(splits)-1))]
            
        return out
    
    def EncodeFrames(self):
        return None
    
    def ExtractFrames():
        return None

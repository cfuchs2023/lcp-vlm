# -*- coding: utf-8 -*-
from .hmdb51 import HMDB51
from .ucf101 import UCF101
from .kinetics400 import Kinetics400
from .utils import FeaturesLoader

#%%
dataloader_dict = {"UCF101":FeaturesLoader, 
                   "hmdb51":HMDB51, 
                   "Kinetics400":Kinetics400,
                   'sun397':FeaturesLoader,
                   'Food101':FeaturesLoader,
                   'dtd':FeaturesLoader,
                   'OxfordPets':FeaturesLoader,
                   'eurosat':FeaturesLoader,
                   'StanfordCars':FeaturesLoader,
                   'Caltech101':FeaturesLoader,
                   'Flower102':FeaturesLoader,
                   'fgvc_aircraft':FeaturesLoader,
                   'imagenet':FeaturesLoader
                   #'UCF101':FeaturesLoader
                   }
                   

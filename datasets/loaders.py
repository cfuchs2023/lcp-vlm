# -*- coding: utf-8 -*-
from .utils import FeaturesLoader

#%%
dataloader_dict = {"UCF101":FeaturesLoader, 
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
                   

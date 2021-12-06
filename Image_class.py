#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import imptools as tool

class ProcessedImage:
    def __init__(self, original_AFM):
        self.original_image = original_AFM
        self.binarized_image = None
        self.calibrated_image = None
        self.skeltone_image = None
        self.label_image = None
        self.fiber_positions = None  
 
#ここから下は、各画像処理クラスが完成してから
    def heights_skeltone(self):
        '''
        画像中の細線部の高さ全部拾ってくるメソッド
        '''
        skeltone_posi = self.skeltone_image.astype(bool)
        heights_of_skeltone_lines = self.calibrated_image[skeltone_posi]
        return heights_of_skeltone_lines
    
#     def all_length_distribution(self):
#         '''
#         obtaine length distribution including isolated and overlapped CNF
#         raise error if self.label_image == None 
#         '''
#         n_label = np.max(self.label_image)
#         for label in range(n_label):
#             all_length_distributions = tool.get_length
#         return all_length_distributions
    
#     def isolated_length_distribution(self):
#         '''
#         obtaine length distribution from isolated CNF
#         '''
#         return isolated_length_distribution
    
#     def average_length_distribution(self):
#         return average_length_distribution


# In[ ]:


class Fiber:
    def __init__(self):
        self.xcontour = None
        self.ycontour = None


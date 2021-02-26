# -*- coding: utf-8 -*-
"""
Created on Thu Feb 18 16:36:40 2021

@author: tusha
"""

from files.pipeline import final
obj=final()

def test_load_audio(random_input):
    ''' This function test the load_audio function
        Parameters:
            random_input(int):index of audio file 
        Returns:
            Boolean: Return True if load_audio function works as expected else False
    '''
    raw_audio=obj.load_audio(random_input)
    assert(len(raw_audio)<=570776)
    return True
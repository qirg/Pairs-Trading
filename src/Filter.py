import numpy as np
import os
import requests

class Pairs: # call ony if you need an updated pairs list

    def __init__(self):
        pass

    def pull_data(self):
        '''
        :return: nothing, pull data from saved files
        '''
        pass

    def cointegrated(self): # save to file
        '''
        :return: dict, check for cointegration - O(n^2)
        '''
        pass

class Filters:

    def __init__(self):
        pass

    def market_cap(self):
        pass

    def volume(self):
        pass

    def implied_vol(self):
        pass

    def momentum(self):
        pass


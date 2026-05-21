import numpy as np

'''
fChannel.py
'''


def myf_channel_fig3(fixed_channel, param_channel):

    ### Parameters ###

    N = param_channel["N"]

    K = param_channel["K"]


    ### Functions ###

    channel = {}
    
    channel["H"] = np.copy(fixed_channel[N][0:K, :])

    return channel

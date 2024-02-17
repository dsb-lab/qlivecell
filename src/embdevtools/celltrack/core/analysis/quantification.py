def quantify_channels(CT):
    import numpy as np
    quantifications = [[] for ch in CT.channels]
    for cell in CT.jitcells:
        zc = int(cell.centers[0][0])
        zcid = cell.zs[0].index(zc)

        mask = cell.masks[0][zcid]
        mask[mask < 0] = 0
        mask[:,0][mask[:,0]>=CT.hyperstack.shape[-2]] = CT.hyperstack.shape[2]-1
        mask[:,1][mask[:,1]>=CT.hyperstack.shape[-1]] = CT.hyperstack.shape[-1]-1
        for ch_id, ch in enumerate(CT.channels):
            stack = CT.hyperstack[0, zc, ch, :, :]
            quantifications[ch_id].append(np.mean(stack[mask[:,0], mask[:,1]]))
    
    return quantifications

def plot_channel_quantification_bar(CT, channel_labels=None):
    colors = ["yellow", "magenta", "green", "blue"]
    quantifications = quantify_channels(CT)
    import numpy as np
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    if channel_labels is None:
        channel_labels = [i for i in range(len(quantifications))]
    ax.bar(channel_labels, np.mean(quantifications, axis=1), yerr=np.std(quantifications, axis=1), capsize=10,  color=colors[:len(quantifications)])
    ax.set_ylabel("mean nuclear fluorescence")
    plt.show()
    

def plot_channel_quantification_hist(CT, **kwargs):
    colors = ["yellow", "magenta", "green", "blue"]
    quantifications = quantify_channels(CT)
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    if "channel_labels" not in kwargs.keys():
        channel_labels = [i for i in range(len(quantifications))]
    else: 
        channel_labels = kwargs["channel_labels"]
    
    if "bins" not in kwargs.keys():
        kwargs["bins"]=50
    
    if "log" not in kwargs.keys():
        kwargs["log"]=False
    
    fig, ax = plt.subplots(len(channel_labels), 1, figsize=(10, 5*len(channel_labels)), sharex=True)
    for ch in range(len(channel_labels)):
        if kwargs["log"]:
            ax[ch].set_xscale('log')
            bins=np.logspace(np.log10(np.min(quantifications)),np.log10(np.max(quantifications)), kwargs["bins"])
        else: 
            bins = kwargs["bins"]

        ax[ch].hist(quantifications[ch],  color=colors[ch], bins=bins, density=True, label=channel_labels[ch])
        ax[ch].legend()
        
    ax[-1].set_xlabel("mean nuclear fluorescence")
    plt.show()
    
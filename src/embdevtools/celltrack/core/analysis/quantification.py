import numpy as np
from scipy.interpolate import splrep, BSpline
from scipy.signal import butter, lfilter

# From https://stackoverflow.com/a/12233959
def butter_bandpass(lowcut, highcut, fs, order=5):
    return butter(order, [lowcut, highcut], fs=fs, btype='band')

# From https://stackoverflow.com/a/12233959
def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def extract_fluoro(CT):
    results = {}
    results["centers"] = []
    results["labels"] = []
    results["centers_px"] = []
    results["slices"] = CT.hyperstack.shape[1]
    results["masks"] = []
    results["zres"] = CT.metadata["Zresolution"]
    _ch = []
    for ch in range(CT.hyperstack.shape[2]):
        results["channel_{}".format(ch)] = []
        _ch.append([])
    for cell in CT.jitcells:
        for zid, z in enumerate(cell.zs[0]):
            mask = cell.masks[0][zid]
            
            if z == cell.centers[0][0]:
                results["masks"].append(mask)
            
            for ch in range(CT.hyperstack.shape[2]):
                img = CT.hyperstack[0, z, ch]
                _ch[ch].append(np.mean(img[mask[:,1], mask[:,0]]))

        for ch in range(CT.hyperstack.shape[2]):
            results["channel_{}".format(ch)].append(np.mean(_ch[ch]))
            del _ch[ch][:]

        zres = CT.metadata["Zresolution"]
        xyres = CT.metadata["XYresolution"]
        results["centers_px"].append(cell.centers[0])
        results["centers"].append(cell.centers[0]*[zres, xyres, xyres])
        results["labels"].append(cell.label + 1)

    for i, j in results.items():
        results[i] = np.array(j)
  
    return results


import numpy as np
from scipy.optimize import curve_fit

def linear_decay(z, slope, intercept):
    return slope * z + intercept

def get_intenity_profile(CT, ch):

    image_stack = CT.hyperstack[0,:,ch]

    intensity_per_z = np.zeros(CT.slices)
    intensity_per_z_n = np.zeros(CT.slices)

    for cell in CT.jitcells:
        zc = int(cell.centers[0][0])
        zcid = cell.zs[0].index(zc)

        msk = cell.masks[0][zcid]
        img = image_stack[zc]
        intensity = np.mean(img[msk[:, 1], msk[:, 0]])
        
        intensity_per_z_n[zc] += 1
        intensity_per_z[zc] += intensity


    zs = np.where(intensity_per_z_n!=0)[0]
    data_z = intensity_per_z[zs] / intensity_per_z_n[zs]
    data_z_filled = []
    zs_filled = []
    for z in range(zs[0], zs[-1]+1):
        if z in zs:
            zid = np.where(zs==z)[0][0]
            data_z_filled.append(data_z[zid])
            zs_filled.append(z)
        else:
            if (z + 1 in zs):
                zid = np.where(zs==z+1)[0][0]
                data_z_filled.append(np.mean([data_z[zid], data_z_filled[-1]])) 
                zs_filled.append(z)
            else: 
                data_z_filled.append(data_z_filled[-1])
                zs_filled.append(z)
    # Measure intensity profile along the z-axis
    intensity_profile = np.array(data_z_filled)

    # Define z-axis positions
    z_positions = np.array(zs_filled)

    # Fit linear decay to intensity profile
    popt, _ = curve_fit(linear_decay, z_positions, intensity_profile)
    slope, intercept = popt

    # Correct intensity
    image_stack = image_stack.astype("float32")
    correction_function = []
    for i in range(image_stack.shape[0]):
        if i not in zs:
            correct_val = linear_decay(i, slope, intercept)
        else:
            zid = np.where(z_positions==i)[0][0]
            correct_val = intensity_profile[zid]
        correction_function.append(correct_val)
    
    return correction_function, intensity_profile, z_positions

def correct_drift(results, ch=0, plotting=False):

    data = results["channel_{}".format(ch)]

    z_order = np.argsort(results["centers_px"][:, 0])
    centers_ordered = np.array(results["centers_px"])[z_order]
    data = np.array(data)[z_order]

    data_z = []
    zs = []
    for z in range(int(max(centers_ordered[:,0]))):
        ids = np.where(centers_ordered[:,0] == z)
        d = data[ids]
        if len(d) == 0: continue
        zs.append(z)
        data_z.append(np.mean(d))

    data_z_filled = []
    for z in range(results["slices"]):
        if z in zs:
            zid = zs.index(z)
            data_z_filled.append(data_z[zid])
        else:
            if z == 0:
               data_z_filled.append(data_z[0])  
            if (z + 1 in zs):
                zid = zs.index(z+1)
                data_z_filled.append(np.mean([data_z[zid], data_z_filled[-1]])) 
            else: 
                data_z_filled.append(data_z_filled[-1])
    
    drift_correction = butter_bandpass_filter(data_z_filled, 0.1*results["zres"], results["zres"]/2.01, results["zres"], order=6)
    drift_correction += np.mean(data_z)

    if plotting:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.plot(data_z)
        ax.plot(drift_correction)
        plt.show()
    
    correct_factor = drift_correction - data_z_filled
    return correct_factor, data_z

def quantify_channels(CT):
    import numpy as np

    quantifications = [[] for ch in CT.channels]
    for cell in CT.jitcells:
        zc = int(cell.centers[0][0])
        zcid = cell.zs[0].index(zc)

        mask = cell.masks[0][zcid]
        mask[mask < 0] = 0
        mask[:, 0][mask[:, 0] >= CT.hyperstack.shape[-2]] = CT.hyperstack.shape[2] - 1
        mask[:, 1][mask[:, 1] >= CT.hyperstack.shape[-1]] = CT.hyperstack.shape[-1] - 1
        for ch_id, ch in enumerate(CT.channels):
            stack = CT.hyperstack[0, zc, ch, :, :]
            quantifications[ch_id].append(np.mean(stack[mask[:, 0], mask[:, 1]]))

    return quantifications


def plot_channel_quantification_bar(CT, channel_labels=None):
    colors = ["yellow", "magenta", "green", "blue"]
    quantifications = quantify_channels(CT)
    import matplotlib.pyplot as plt
    import numpy as np

    fig, ax = plt.subplots()
    if channel_labels is None:
        channel_labels = [i for i in range(len(quantifications))]
    ax.bar(
        channel_labels,
        np.mean(quantifications, axis=1),
        yerr=np.std(quantifications, axis=1),
        capsize=10,
        color=colors[: len(quantifications)],
    )
    ax.set_ylabel("mean nuclear fluorescence")
    plt.show()


def plot_channel_quantification_hist(CT, **kwargs):
    colors = ["yellow", "magenta", "green", "blue"]
    quantifications = quantify_channels(CT)

    import matplotlib.pyplot as plt
    import numpy as np

    if "channel_labels" not in kwargs.keys():
        channel_labels = [i for i in range(len(quantifications))]
    else:
        channel_labels = kwargs["channel_labels"]

    if "bins" not in kwargs.keys():
        kwargs["bins"] = 50

    if "log" not in kwargs.keys():
        kwargs["log"] = False

    fig, ax = plt.subplots(
        len(channel_labels), 1, figsize=(10, 5 * len(channel_labels)), sharex=True
    )
    for ch in range(len(channel_labels)):
        if kwargs["log"]:
            ax[ch].set_xscale("log")
            bins = np.logspace(
                np.log10(np.min(quantifications)),
                np.log10(np.max(quantifications)),
                kwargs["bins"],
            )
        else:
            bins = kwargs["bins"]

        ax[ch].hist(
            quantifications[ch],
            color=colors[ch],
            bins=bins,
            density=True,
            label=channel_labels[ch],
        )
        ax[ch].legend()

    ax[-1].set_xlabel("mean nuclear fluorescence")
    plt.show()

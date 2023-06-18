from skimage.segmentation import morphological_chan_vese, checkerboard_level_set
from .core.utils_ES import *
import matplotlib.pyplot as plt
import numpy as np

class EmbryoSegmentation():
    def __init__(self, IMGS, ksize=5, ksigma=3, binths=8, apply_biths_to_zrange_only=False, checkerboard_size=6, num_inter=100, smoothing=5, trange=None, zrange=None, mp_threads=None):
        
        self.times  = IMGS.shape[0]
        self.slices = IMGS.shape[1]

        if trange is None: self.trange=range(self.times)
        else: self.trange=trange
        if zrange is None: self.zrange=range(self.slices)
        else:self.zrange=zrange

        self.Emb  = [[] for t in self.trange]
        self.Back = [[] for t in self.trange]
        self.LS   = [[] for t in self.trange]
        self.Embmask  = [[] for t in self.trange]
        self.Backmask = [[] for t in self.trange]
        
        if mp_threads == "all": self._threads=mp.cpu_count()-1
        else: self._threads = mp_threads

        self.ksize=ksize
        self.ksigma=ksigma
        if type(binths) == list: 
            if len(binths) == 2:
                self.binths = np.zeros(self.slices) 
                if apply_biths_to_zrange_only: self.binths[self.zrange] = np.linspace(binths[0], binths[1], num=len(self.zrange))
                else: self.binths = np.linspace(binths[0], binths[1], num=self.slices)
            else: self.binths = binths
        else: self.binths = [binths for i in range(self.slices)]
        self.checkerboard_size=checkerboard_size
        self.num_inter=num_inter
        self.smoothing=smoothing
    
    def __call__(self, IMGS):
        seg_embryo_params=(self.ksize, self.ksigma, self.checkerboard_size, self.smoothing, self.num_inter)
        for tid, t in enumerate(self.trange):
            print("t =", t)
            if self._threads is None:
                results=[]
                for zid,z in enumerate(range(self.slices)):
                    result = self.compute_emb_masks_z(IMGS[t][z], z, tid, zid)
                    results.append(result)
            else:
                TASKS = [(compute_emb_masks_z, ((IMGS[t][z], z, tid, zid, self.binths[z], seg_embryo_params))) for zid,z in enumerate(self.zrange)]
                results = multiprocess(self._threads, worker, TASKS)

            results.sort(key=lambda x: x[1])
            for result in results:
                tid, zid, ls, emb, back, embmask, backmask = result
                if len(ls)!=0:
                    self.LS[tid].append(ls)

                    self.Emb[tid].append(emb) 
                    self.Back[tid].append(back)
                
                self.Embmask[tid].append(embmask)
                self.Backmask[tid].append(backmask)
        return

    def plot_segmentation(self, ts, zs, plot_background=True, extra_IMGS=None, extra_title=''):
                
        if not isinstance(ts, list): ts=[ts]
        if not isinstance(zs, list): zs=[zs]

        naxes = 1
        if plot_background: naxes+=1
        if extra_IMGS is not None: naxes+=1
        
        fig, ax = plt.subplots(len(ts),naxes,figsize=(18, 6*len(ts)))
        
        ids = np.arange(len(ax.flatten())).reshape(np.shape(ax))
        ax = ax.flatten()
        for id, t in enumerate(ts):
            z=zs[id]

            if t not in self.trange: continue
            if z not in self.zrange: continue

            tid = self.trange.index(t)
            zid = self.zrange.index(z)
            
            emb_segment = np.array(self.Emb[tid][zid])
            background  = np.array(self.Back[tid][zid])
            embmask     = np.array(self.Embmask[tid][zid])
            backmask    = np.array(self.Backmask[tid][zid])
            ls          = np.array(self.LS[tid][zid])
            
            try: id0 = ids[id,0]
            except: id0 = 0
            ax[id0].imshow(emb_segment)
            ax[id0].set_axis_off()
            ax[id0].contour(ls, [0.5], colors='r')
            #ax[id0].scatter(embmask[:,0], embmask[:,1], s=0.1, c='red', alpha=0.1)
            ax[id0].set_title("Morphological ACWE - mask", fontsize=12)

            if plot_background:
                try: id1 = ids[id,1]
                except: id1 = 1
                ax[id1].imshow(background)
                ax[id1].set_axis_off()
                ax[id1].contour(ls, [0.5], colors='r')
                #ax[id1].scatter(backmask[:,0], backmask[:,1], s=0.1, c='red', alpha=0.1)
                ax[id1].set_title("Morphological ACWE - background", fontsize=12)

            if extra_IMGS is not None: 
                try: id2 = ids[id,-1]
                except: id2 = -1
                ax[id2].imshow(extra_IMGS[t][z])
                ax[id2].set_axis_off()
                ax[id2].contour(ls, [0.5], colors='r')
                ax[id2].set_title(extra_title, fontsize=12)
                
        fig.tight_layout()
        plt.show()



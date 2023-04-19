from skimage.segmentation import morphological_chan_vese, checkerboard_level_set
from utils_ES import *
import matplotlib.pyplot as plt

class EmbryoSegmentation():
    def __init__(self, IMGS, ksize=5, ksigma=3, binths=8, apply_biths_to_zrange_only=False, checkerboard_size=6, num_inter=100, smoothing=5, trange=None, zrange=None, mp_threads=None):
        
        self.times  = IMGS.shape[0]
        self.slices = IMGS.shape[1]
        self.Emb  = [[] for t in trange]
        self.Back = [[] for t in trange]
        self.LS   = [[] for t in trange]
        self.Embmask  = [[] for t in trange]
        self.Backmask = [[] for t in trange]
        
        if mp_threads == "all": self._threads=mp.cpu_count()-1
        else: self._threads = mp_threads

        if trange is None: self.trange=range(self.times)
        else: self.trange=trange
        if zrange is None: self.zrange=range(self.slices)
        else:self.zrange=zrange
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
                    t = self.trange[tid]
                    z = self.zrange[zid]
                    self.LS[t].append(ls)

                    self.Emb[t].append(emb) 
                    self.Back[t].append(back)
                
                self.Embmask[t].append(embmask)
                self.Backmask[t].append(backmask)
        return

    def plot_segmentation(self, t, z, extra_IMGS=None):
        if t not in self.trange: raise Exception("t not in selected time range") 
        if z not in self.zrange: raise Exception("z not in selected slice range") 
        tid = self.trange.index(t)
        zid = self.zrange.index(z)
        
        emb_segment = np.array(self.Emb[tid][zid])
        background  = np.array(self.Back[tid][zid])
        embmask     = np.array(self.Embmask[tid][zid])
        backmask    = np.array(self.Backmask[tid][zid])
        ls          = np.array(self.LS[tid][zid])
        
        if extra_IMGS is None: fig, ax = plt.subplots(1,2,figsize=(12, 6))
        else: 
            fig, ax = plt.subplots(1,3,figsize=(18, 6))
            ax[2].imshow(extra_IMGS[t][z])
            ax[2].set_axis_off()
            ax[2].contour(ls, [0.5], colors='r')
            ax[2].set_title("nuclear channel", fontsize=12)
        ax[0].imshow(emb_segment)
        ax[0].set_axis_off()
        ax[0].contour(ls, [0.5], colors='r')
        #ax[0].scatter(embmask[:,0], embmask[:,1], s=0.1, c='red', alpha=0.1)
        ax[0].set_title("Morphological ACWE - mask", fontsize=12)

        ax[1].imshow(background)
        ax[1].set_axis_off()
        ax[1].contour(ls, [0.5], colors='r')
        #ax[1].scatter(backmask[:,0], backmask[:,1], s=0.1, c='red', alpha=0.1)
        ax[1].set_title("Morphological ACWE - background", fontsize=12)

        #fig.tight_layout()
        plt.show()



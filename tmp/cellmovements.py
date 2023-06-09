    # def compute_cell_movement(self, movement_computation_method):
    #     for cell in self.cells:
    #         cell.compute_movement(self._cdaxis, movement_computation_method)

    # def compute_mean_cell_movement(self):
    #     nrm = np.zeros(self.times-1)
    #     self.cell_movement = np.zeros(self.times-1)
    #     for cell in self.cells:
    #         time_ids = np.array(cell.times)[:-1]
    #         nrm[time_ids]+=np.ones(len(time_ids))
    #         self.cell_movement[time_ids]+=cell.disp
    #     self.cell_movement /= nrm
            
    # def cell_movement_substract_mean(self):
    #     for cell in self.cells:
    #         new_disp = []
    #         for i,t in enumerate(cell.times[:-1]):
    #             new_val = cell.disp[i] - self.cell_movement[t]
    #             new_disp.append(new_val)
    #         cell.disp = new_disp

    # def plot_cell_movement(self
    #                      , label_list=None
    #                      , plot_mean=True
    #                      , substract_mean=None
    #                      , plot_tracking=True
    #                      , plot_layout=None
    #                      , plot_overlap=None
    #                      , masks_cmap=None
    #                      , movement_computation_method=None):
        
    #     if movement_computation_method is None: movement_computation_method=self._movement_computation_method
    #     else: self._movement_computation_method=movement_computation_method
    #     if substract_mean is None: substract_mean=self._mscm
    #     else: self._mscm=substract_mean
        
    #     self.compute_cell_movement(movement_computation_method)
    #     self.compute_mean_cell_movement()
    #     if substract_mean:
    #         self.cell_movement_substract_mean()
    #         self.compute_mean_cell_movement()

    #     ymax  = max([max(cell.disp) if len(cell.disp)>0 else 0 for cell in self.cells])+1
    #     ymin  = min([min(cell.disp) if len(cell.disp)>0 else 0 for cell in self.cells])-1

    #     if label_list is None: label_list=list(copy(self.unique_labels))
        
    #     used_markers = []
    #     used_styles  = []
    #     if hasattr(self, "fig_cellmovement"):
    #         if plt.fignum_exists(self.fig_cellmovement.number):
    #             firstcall=False
    #             self.ax_cellmovement.cla()
    #         else:
    #             firstcall=True
    #             self.fig_cellmovement, self.ax_cellmovement = plt.subplots(figsize=(10,10))
    #     else:
    #         firstcall=True
    #         self.fig_cellmovement, self.ax_cellmovement = plt.subplots(figsize=(10,10))
        
    #     len_cmap = len(self._label_colors)
    #     len_ls   = len_cmap*len(PLTMARKERS)
    #     countm   = 0
    #     markerid = 0
    #     linestyleid = 0
    #     for cell in self.cells:
    #         label = cell.label
    #         if label in label_list:
    #             c  = self._label_colors[self._labels_color_id[label]]
    #             m  = PLTMARKERS[markerid]
    #             ls = PLTLINESTYLES[linestyleid]
    #             if m not in used_markers: used_markers.append(m)
    #             if ls not in used_styles: used_styles.append(ls)
    #             tplot = [cell.times[i]*self._tstep for i in range(1,len(cell.times))]
    #             self.ax_cellmovement.plot(tplot, cell.disp, c=c, marker=m, linewidth=2, linestyle=ls,label="%d" %label)
    #         countm+=1
    #         if countm==len_cmap:
    #             countm=0
    #             markerid+=1
    #             if markerid==len(PLTMARKERS): 
    #                 markerid=0
    #                 linestyleid+=1
    #     if plot_mean:
    #         tplot = [i*self._tstep for i in range(1,self.times)]
    #         self.ax_cellmovement.plot(tplot, self.cell_movement, c='k', linewidth=4, label="mean")
    #         leg_patches = [Line2D([0], [0], color="k", lw=4, label="mean")]
    #     else:
    #         leg_patches = []

    #     label_list_lastdigit = [int(str(l)[-1]) for l in label_list]
    #     for i, col in enumerate(self._label_colors):
    #         if i in label_list_lastdigit:
    #             leg_patches.append(Line2D([0], [0], color=col, lw=2, label=str(i)))

    #     count = 0
    #     for i, m in enumerate(used_markers):
    #         leg_patches.append(Line2D([0], [0], marker=m, color='k', label="+%d" %count, markersize=10))
    #         count+=len_cmap

    #     count = 0
    #     for i, ls in enumerate(used_styles):
    #         leg_patches.append(Line2D([0], [0], linestyle=ls, color='k', label="+%d" %count, linewidth=2))
    #         count+=len_ls

    #     self.ax_cellmovement.set_ylabel("cell movement")
    #     self.ax_cellmovement.set_xlabel("time (min)")
    #     self.ax_cellmovement.xaxis.set_major_locator(MaxNLocator(integer=True))
    #     self.ax_cellmovement.legend(handles=leg_patches, bbox_to_anchor=(1.04, 1))
    #     self.ax_cellmovement.set_ylim(ymin,ymax)
    #     self.fig_cellmovement.tight_layout()
        
    #     if firstcall:
    #         if plot_tracking:
    #             self.plot_tracking(windows=1, cell_picker=True, plot_layout=plot_layout, plot_overlap=plot_overlap, masks_cmap=masks_cmap, mode="CM")
    #         else: plt.show()

    # def _select_cells(self
    #                 , plot_layout=None
    #                 , plot_overlap=None
    #                 , masks_cmap=None):
        
    #     self.plot_tracking(windows=1, cell_picker=True, plot_layout=plot_layout, plot_overlap=plot_overlap, masks_cmap=masks_cmap, mode="CP")
    #     self.PACP.CP.stopit()
    #     labels = copy(self.PACP.label_list)
    #     return labels

    
    # def plot_masks3D_Imagej(self
    #                       , verbose=False
    #                       , cell_selection=False
    #                       , plot_layout=None
    #                       , plot_overlap=None
    #                       , masks_cmap=None
    #                       , keep=True
    #                       , color=None
    #                       , channel_name=""):
        
    #     self.save_masks3D_stack(cell_selection, plot_layout=plot_layout, plot_overlap=plot_overlap, masks_cmap=masks_cmap, color=color, channel_name=channel_name)
    #     file=self.embcode+"_masks"+channel_name+".tiff"
    #     pth=self.path_to_save
    #     fullpath = pth+file
        
    #     if verbose:
    #         subprocess.run(['/opt/Fiji.app/ImageJ-linux64', '--ij2', '--console', '-macro', '/home/pablo/Desktop/PhD/projects/CellTracking/utils/imj_3D.ijm', fullpath])
    #     else:
    #         subprocess.run(['/opt/Fiji.app/ImageJ-linux64', '--ij2', '--console', '-macro', '/home/pablo/Desktop/PhD/projects/CellTracking/utils/imj_3D.ijm', fullpath], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    #     if not keep:
    #         subprocess.run(["rm", fullpath])
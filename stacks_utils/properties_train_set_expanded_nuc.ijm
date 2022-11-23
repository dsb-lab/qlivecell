mypath = "/home/pablo/Desktop/PhD/projects/Data/blastocysts/movies/2h_claire_ERK-KTR_MKATE2/cell_tracking/training_set_expanded_nuc/"
imagelist=getFileList(mypath);
for (j=0; j<imagelist.length; j++) {
    open(mypath+imagelist[j]);
    Stack.setXUnit("µm");
    Stack.setYUnit("µm");
    Stack.setZUnit("µm");
    run("Properties...", "channels=1 slices=1 frames=1 pixel_width=0.2767553 pixel_height=0.2767553 voxel_depth=2");
    saveAs(mypath+imagelist[j]);
    close();
}

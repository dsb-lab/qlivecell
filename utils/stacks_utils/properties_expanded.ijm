mypath = "/home/pablo/Desktop/PhD/projects/Data/blastocysts/movies/2h_claire_ERK-KTR_MKATE2/expanded_stacks/"
filelist = getFileList(mypath)
for (i=0; i<filelist.length; i++) {
  imagelist=getFileList(mypath+filelist[i]+"/");
  for (j=0; j<imagelist.length; j++) {
    open(mypath+filelist[i]+imagelist[j]);
    Stack.setXUnit("µm");
    Stack.setYUnit("µm");
    Stack.setZUnit("µm");
    run("Properties...", "channels=2 slices=1 frames=1 pixel_width=0.2767553 pixel_height=0.2767553 voxel_depth=2");
    saveAs(mypath+filelist[i]+imagelist[j]);
    close();
  }
}

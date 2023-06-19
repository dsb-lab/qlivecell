fullpath = getArgument()
//open(fullpath);
run("Bio-Formats Windowless Importer", "open="+fullpath);
imageName=getTitle(); 
//print(imageName)
selectImage(imageName)
run("3D Viewer");
call("ij3d.ImageJ3DViewer.setCoordinateSystem", "false");
call("ij3d.ImageJ3DViewer.add", imageName , "None", imageName, "0", "true", "true", "true", "1", "0");
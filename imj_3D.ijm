fullpath = getArgument()
run("Bio-Formats Windowless Importer", "open="+fullpath);
//open(fullpath);
imageName=getTitle(); 
//print(imageName)
selectImage(imageName)
run("3D Viewer");
call("ij3d.ImageJ3DViewer.setCoordinateSystem", "false");
call("ij3d.ImageJ3DViewer.add", "Lineage_2hr_082219_p2_E375_masks.tiff", "None", "Lineage_2hr_082219_p2_E375_masks.tiff", "0", "true", "true", "true", "1", "0");
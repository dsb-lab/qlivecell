include("utils_ct.jl")

println("#                                         #")
println("#                                         #")
println("#   Expand completely stacks into images  #")
println("#                                         #")
println("#                                         #")

if !in("expanded_stacks_nuc", readdir("/home/pablo/Desktop/PhD/projects/Data/blastocysts/movies/2h_claire_ERK-KTR_MKATE2"))
    mkdir("/home/pablo/Desktop/PhD/projects/Data/blastocysts/movies/2h_claire_ERK-KTR_MKATE2/expanded_stacks_nuc")
end
emblist = readdir("/home/pablo/Desktop/PhD/projects/Data/blastocysts/movies/2h_claire_ERK-KTR_MKATE2/original")
for i=1:length(emblist)
    embcode=emblist[i]
    embcode=embcode[1:end-4]
    if !in(embcode,readdir("/home/pablo/Desktop/PhD/projects/Data/blastocysts/movies/2h_claire_ERK-KTR_MKATE2/expanded_stacks_nuc"))
        mkdir("/home/pablo/Desktop/PhD/projects/Data/blastocysts/movies/2h_claire_ERK-KTR_MKATE2/expanded_stacks_nuc/$embcode")
    end
    raw_data = TiffImages.load("/home/pablo/Desktop/PhD/projects/Data/blastocysts/movies/2h_claire_ERK-KTR_MKATE2/original/$embcode.tif", verbose=false)
    # Extract stack
    IMG1=extract_stacks(raw_data, colors=2, used_colors=[2])
    IMG =extract_stacks(raw_data, colors=2, used_colors=[1])
    IMGs = [IMG, IMG1]
    ## Reduce image sizes
    nt = size(IMG)[4]
    c=2
    for i=1:nt
        for z=1:size(IMG)[3]
            imgname=join(["t$i","z$z", "c$c"])
            FileIO.save("/home/pablo/Desktop/PhD/projects/Data/blastocysts/movies/2h_claire_ERK-KTR_MKATE2/expanded_stacks_nuc/$embcode/$imgname.tif", IMGs[c][:,:,z,i]);
        end
    end
end

## Give the properties to the images
println("#                                         #")
println("#                                         #")
println("#     Restore props of created images     #")
println("#                                         #")
println("#                                         #")
run(`/opt/Fiji.app/ImageJ-linux64 -batch properties_train_set_expanded_nuc.ijm`)
println("#                                         #")
println("#                                         #")
println("###########################################")


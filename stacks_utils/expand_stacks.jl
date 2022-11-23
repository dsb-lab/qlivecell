include("utils_ct.jl")

println("#                                         #")
println("#                                         #")
println("#   Expand completely stacks into images  #")
println("#                                         #")
println("#                                         #")

if !in("expanded_stacks_ch", readdir("/home/pablo/Desktop/PhD/projects/Data/blastocysts/movies/2h_claire_ERK-KTR_MKATE2"))
    mkdir("/home/pablo/Desktop/PhD/projects/Data/blastocysts/movies/2h_claire_ERK-KTR_MKATE2/expanded_stacks")
end
emblist = readdir("/home/pablo/Desktop/PhD/projects/Data/blastocysts/movies/2h_claire_ERK-KTR_MKATE2/original")
for i=1:length(emblist)
    embcode=emblist[i]
    embcode=embcode[1:end-4]
    if !in(embcode,readdir("/home/pablo/Desktop/PhD/projects/Data/blastocysts/movies/2h_claire_ERK-KTR_MKATE2/expanded_stacks"))
        mkdir("/home/pablo/Desktop/PhD/projects/Data/blastocysts/movies/2h_claire_ERK-KTR_MKATE2/expanded_stacks/$embcode")
    end
    raw_data = TiffImages.load("/home/pablo/Desktop/PhD/projects/Data/blastocysts/movies/2h_claire_ERK-KTR_MKATE2/original/$embcode.tif", verbose=false)
    # Extract stack
    IMG1=extract_stacks(raw_data, colors=2, used_colors=[2])
    IMG =extract_stacks(raw_data, colors=2, used_colors=[1])
    currentIMG=zeros(size(IMG)[1],size(IMG)[2],2)
    ## Reduce image sizes
    nt = size(IMG)[4]
    for i=1:nt
        for z=1:size(IMG)[3]
            currentIMG[:,:,1] .= IMG[:,:,z,i]
            currentIMG[:,:,2] .= IMG1[:,:,z,i]
            imgname=join(["t$i","z$z"])
            FileIO.save("/home/pablo/Desktop/PhD/projects/Data/blastocysts/movies/2h_claire_ERK-KTR_MKATE2/expanded_stacks/$embcode/$imgname.tif", currentIMG);
        end
    end
end

## Give the properties to the images
println("#                                         #")
println("#                                         #")
println("#     Restore props of created images     #")
println("#                                         #")
println("#                                         #")
run(`/opt/Fiji.app/ImageJ-linux64 -batch properties_expanded.ijm`)
println("#                                         #")
println("#                                         #")
println("###########################################")


using ImageTransformations
using TiffImages
using FileIO
function extract_stacks( raw_data
    ; datadims=(512,512)
    , used_colors=[1,2]
    , used_stacks=30
    , colors=2
    , times=24
    , total_stacks=30
    , norm=false)


    stp=total_stacks*colors
    IMG = zeros(datadims[1],datadims[2],total_stacks,times)
    I=zeros(datadims[1],datadims[2],total_stacks)
    for t=1:times
        I.=zeros(datadims[1],datadims[2],total_stacks)
        for c in used_colors
            jj=c-1
            I .+= Array{Float64,3}(raw_data[:,:,((t-1)*stp+1)+jj:colors:t*stp + jj])
        end
        if norm
            IMG[:,:,:,t].= I./maximum(I)
        else
            IMG[:,:,:,t].= I
        end
    end
    return IMG
end

"""
    This function reduces the resolucion of the images within a stack to reddim
"""
function img_reduction(IMG, reddim::Int64; slices=30, times=24, norm=true)
    pre_IMG_Re=zeros(reddim, reddim, slices, times)
    Threads.@threads for i=1:slices
        for j=1:times
            pre_IMG_Re[:,:,i,j].=imresize(IMG[:,:,i,j], (reddim, reddim))
            if norm
                if maximum(pre_IMG_Re[:,:,i,j])==0
                    nothing
                else
                    pre_IMG_Re[:,:,i,j]./=maximum(pre_IMG_Re[:,:,i,j])
                end
            end
        end
    end
    return pre_IMG_Re
end

function generate_train_set_expanded(ntrain)
    if !in("cell_tracking",readdir("/home/pablo/Desktop/PhD/projects/Data/blastocysts/movies/2h_claire_ERK-KTR_MKATE2/"))
        mkdir("/home/pablo/Desktop/PhD/projects/Data/blastocysts/movies/2h_claire_ERK-KTR_MKATE2/cell_tracking")
    end
    if !in("training_set_expanded",readdir("/home/pablo/Desktop/PhD/projects/Data/blastocysts/movies/2h_claire_ERK-KTR_MKATE2/cell_tracking"))
        mkdir("/home/pablo/Desktop/PhD/projects/Data/blastocysts/movies/2h_claire_ERK-KTR_MKATE2/cell_tracking/training_set_expanded")
    else
        rm("/home/pablo/Desktop/PhD/projects/Data/blastocysts/movies/2h_claire_ERK-KTR_MKATE2/cell_tracking/training_set_expanded", recursive=true)
        mkdir("/home/pablo/Desktop/PhD/projects/Data/blastocysts/movies/2h_claire_ERK-KTR_MKATE2/cell_tracking/training_set_expanded")
    end
    emblist = readdir("/home/pablo/Desktop/PhD/projects/Data/blastocysts/movies/2h_claire_ERK-KTR_MKATE2/original")
    currtraining=1
    while currtraining<(ntrain+1)
        emb = rand(1:length(emblist))
        embcode=emblist[emb]
        embcode=embcode[1:end-4]
        raw_data = TiffImages.load("/home/pablo/Desktop/PhD/projects/Data/blastocysts/movies/2h_claire_ERK-KTR_MKATE2/original/$embcode.tif", verbose=false)
        # Extract stack
        IMG1=extract_stacks(raw_data, colors=2, used_colors=[2])
        IMG =extract_stacks(raw_data, colors=2, used_colors=[1])
        currentIMG=zeros(size(IMG)[1],size(IMG)[2],2)
        nt = size(IMG)[4]
        nz = size(IMG)[3]
        tid = rand(1:nt)
        zid = rand(1:nz)
        currentIMG[:,:,1] .= IMG[:,:,zid,tid]
        currentIMG[:,:,2] .= IMG1[:,:,zid,tid]
        imgname=join(["$currtraining", embcode,"t$tid","z$zid"])
        FileIO.save("/home/pablo/Desktop/PhD/projects/Data/blastocysts/movies/2h_claire_ERK-KTR_MKATE2/cell_tracking/training_set_expanded/$imgname.tif", currentIMG);
        currtraining+=1
    end
end

function generate_train_set_expanded_nuc(ntrain)
    if !in("cell_tracking",readdir("/home/pablo/Desktop/PhD/projects/Data/blastocysts/movies/2h_claire_ERK-KTR_MKATE2/"))
        mkdir("/home/pablo/Desktop/PhD/projects/Data/blastocysts/movies/2h_claire_ERK-KTR_MKATE2/cell_tracking")
    end
    if !in("training_set_expanded_nuc",readdir("/home/pablo/Desktop/PhD/projects/Data/blastocysts/movies/2h_claire_ERK-KTR_MKATE2/cell_tracking"))
        mkdir("/home/pablo/Desktop/PhD/projects/Data/blastocysts/movies/2h_claire_ERK-KTR_MKATE2/cell_tracking/training_set_expanded_nuc")
    else
        rm("/home/pablo/Desktop/PhD/projects/Data/blastocysts/movies/2h_claire_ERK-KTR_MKATE2/cell_tracking/training_set_expanded_nuc", recursive=true)
        mkdir("/home/pablo/Desktop/PhD/projects/Data/blastocysts/movies/2h_claire_ERK-KTR_MKATE2/cell_tracking/training_set_expanded_nuc")
    end
    emblist = readdir("/home/pablo/Desktop/PhD/projects/Data/blastocysts/movies/2h_claire_ERK-KTR_MKATE2/original")
    currtraining=1
    while currtraining<(ntrain+1)
        emb = rand(1:length(emblist))
        embcode=emblist[emb]
        embcode=embcode[1:end-4]
        raw_data =TiffImages.load("/home/pablo/Desktop/PhD/projects/Data/blastocysts/movies/2h_claire_ERK-KTR_MKATE2/original/$embcode.tif", verbose=false)
        # Extract stack
        IMG =extract_stacks(raw_data, colors=2, used_colors=[2])
        currentIMG=zeros(size(IMG)[1],size(IMG)[2])
        nt = size(IMG)[4]
        nz = size(IMG)[3]
        tid = rand(1:nt)
        zid = rand(3:nz-2)
        currentIMG[:,:] .= IMG[:,:,zid,tid]
        imgname=join(["$currtraining", embcode,"t$tid","z$zid"])
        FileIO.save("/home/pablo/Desktop/PhD/projects/Data/blastocysts/movies/2h_claire_ERK-KTR_MKATE2/cell_tracking/training_set_expanded_nuc/$imgname.tif", currentIMG);
        currtraining+=1
    end
    run(`/opt/Fiji.app/ImageJ-linux64 -batch properties_train_set_expanded_nuc.ijm`)
end

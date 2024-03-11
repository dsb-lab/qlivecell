# TODO list for embdevtools repository

## TODO 

- Change of software name
- Add backups
- Functionality with non-squared images 
- Functionality with 16 and 32 bit images

## Bug fixes
-

## Feature requirements
### GUI
- Add multichannel visualization, not only RGB (3 channels). Maybe possible to switch off channels from the GUI?
- Add contrast control both in the arguments and as sliders in the GUI. For multichannel, needs to be possible to control individual channels
- Show confict per times, not only total conflicts
- Visualize conflicting cells. Available as an action from the GUI
- Combine over Z, check continuitity all cell planes, not only from current plane. 
- Double click on visualization mode selects a cell and it's daughters. 
- Add properties to cells for napari plots (Arboretum and normal 3D viewer)

### Segmentation
- Arg-request: Min fluorescence on mask for it to be kept

### Mitosis detection
- Automatic mitosis detection using 3D-unet

### Analysis
- Cell movement
- Cell volume/size
- Cell mixing
- Mito and Apo rates
- Embryo size dynamics
- Embryo shape dynamics
- Fluoro quantification: Nuc + Cyto (Cytodonuts)
- Neighborhood analysis tools

## Ideas
- Need to explore full integration with Napari

## Performance
- Update labels in slower after set batch.
- 2D visualization is slower than the napari basic visualization (e.g. switching planes and times)
- Most of the things are probably not optimal and even the multithreded functions should be optimized

## Image registration 
-  Check posibility of using mid time or last time as reference to reduce blurring 
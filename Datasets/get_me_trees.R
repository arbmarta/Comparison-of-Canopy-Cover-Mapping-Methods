## yuhao.lu@umanitoba.ca

# March 26 2024

# This script will directly download .las file from the City of Winnipeg Open data portal.
# DTM, CHM, and tree top + crown delineation processes are handled using lidR package

library("lidR")
library("terra")
library("rgl")

# set work directory
wd <- "C:\\Users\\luy15\\OneDrive - University of Manitoba\\Project\\UDT_Winnipeg"
setwd(wd)

# make sub-dir for saving outputs
dir.create(paste0(wd, "\\GetMeTrees\\CHM", showWarnings = F))         # for canopy height models
dir.create(paste0(wd, "\\GetMeTrees\\Crowns", showWarnings = F))      # for crown delineations
dir.create(paste0(wd, "\\GetMeTrees\\TreeTops", showWarnings = F))    # for individual tree tops

# set parameters
chm_res <- 0.5 # canopy height model resolution (m)
disk <- 0.2 # disk size for tree segmentation (m)
windown <- 5 # lmf window size (m)
crs_epsg <- 32614 #crs for exporting las, tiff, and shp
crown_min <- 5 # minimal number of points per crown
#

# read building footprint shp
als.list <- read.csv("city_als_list.txt")

##################################################################################################
# Main code below
##################################################################################################


## iterate the following process per .las file ##

for (i in 1: nrow(als.list)) {

  url <- als.list[i,]
  tile <- unlist(strsplit(url, "/"))[5]
  destfile <- paste0(tempdir(),"\\" ,tile)

  print(paste("Processing Tile",tile, "(", i, "/",toString(nrow(als.list)), ") at", Sys.time()))

  # download .las
  las <- download.file(url, destfile, mode = "wb", quiet = T, timeout = 6000)

  # read only xyz, classification, intensity, and returnNumber
  las <- readLAS(destfile, select = "xyzcir")

  # check and clean las file
  # las_check(las, print = F)
  las.clean <- filter_duplicates(las)
  # las_check(las.clean)
  las <- NULL

  ## create canopy height model (CHM)

  # lowest point of this .las tile
  z.min <- min(las.clean@data$Z)

  #### note ####
  # The number of detected trees is correlated to the ws argument.
  # Small windows sizes usually gives more trees, while large windows size
  # generally miss smaller trees that are “hidden” by big trees that contain
  # the highest points in the neighborhood:
  #### #### ####

  # replace every point in the point cloud with a disk of a known radius (e.g. 15 cm)

  las.norm <- normalize_height(las.clean, algorithm = tin())

  # chm based on normalized las
  chm_p2r_05 <- rasterize_canopy(las.norm, chm_res, p2r(subcircle = disk), pkg = "terra")

  # export chm to a tiff image
  terra::writeRaster(chm_p2r_05,
                     paste0(wd,'\\GetMeTrees\\CHM\\chm_', tools::file_path_sans_ext(tile),'.tiff'),
                     overwrite=TRUE)

  # Tree detection routine with a constant windows size of 5m is applied to each CHM pixel
  # reduce window size will lead to more trees being "detected"

  ttops_chm_p2r_05 <- locate_trees(chm_p2r_05, lmf(windown))

  if (nrow(ttops_chm_p2r_05) > 0) {

  # set projection and export tree tops to .shp
  ttops_chm_p2r_05 <- sf::st_zm(ttops_chm_p2r_05)
  # ttops_chm_p2r_05 <- sf::st_transform(ttops_chm_p2r_05, 32614)
  sf::st_crs(ttops_chm_p2r_05) <- crs_epsg

    ttops_chm_p2r_05$treeID <- ttops_chm_p2r_05$treeID <- 1:nrow(ttops_chm_p2r_05)

    ttops_chm_p2r_05$tileID <- tools::file_path_sans_ext(tile)


  sf::st_write(ttops_chm_p2r_05,
               paste0(wd,'\\GetMeTrees\\TreeTops\\ttop_',tools::file_path_sans_ext(tile),'.shp'),
               driver = "ESRI Shapefile",
               delete_layer = T, quiet = T)

  # canopy delineation
  algo <- dalponte2016(chm_p2r_05, ttops_chm_p2r_05)
  sf::st_crs(las.norm) <- st_crs(algo)
  las_trees <- segment_trees(las.norm, algo) # segment point cloud

  # viz individual tree
  # tree <- filter_poi(las_trees, treeID == 2)
  # plot(tree, size = 8, bg = "white")

  # make crowns
  crowns <- crown_metrics(las_trees, func = .stdtreemetrics, geom = "convex")

  # remove trees if no. of points < 5
  crowns <- crowns[crowns$npoints > crown_min,]
  # plot(crowns["convhull_area"], main = "Crown area (convex hull)")
  sf::st_crs(crowns) <- crs_epsg

  # export crowns
  sf::st_write(crowns,
               paste0(wd,'\\GetMeTrees\\Crowns\\crown_',tools::file_path_sans_ext(tile),'.shp'),
               driver = "ESRI Shapefile",
               delete_layer = T, quiet = T)
  }
  # delete .las
  file.remove(destfile)

}

##################################################################################################
# Main code above
##################################################################################################

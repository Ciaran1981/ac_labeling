#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  2 16:46:19 2023

@author: Ciaran Robb

Functions taken from my own lib (https://github.com/Ciaran1981/geospatial-learn)
for the sake of convenience should you not wish to install it comment out the
geo... imports and uncomment the src (though you will have to install the deps)
"""
import geopandas as gpd
from shapely import wkt
import os
from glob2 import glob
#from src import (batch_wms_download, write_vrt, image_thresh,
#                 raster2array, do_ac, ms_snake)
from geospatial_learn.raster import (batch_wms_download, write_vrt, 
                                      image_thresh, raster2array)
from geospatial_learn.shape import ms_snake
from geospatial_learn.utilities import do_ac
from tqdm import tqdm

# Change /skip as appropriate

# Data download 
maindir = os.getcwd()
# The csv you shared with me
geomcsv = ('ErosionMapping.csv')

gdf  = gpd.read_file(geomcsv)
# apply as geometry just so we can use if required 
# btw just moving the column to geometry doesn't work - format must be not right
gdf['geometry'] = gdf.wkt_geom.apply(wkt.loads)
#dump
gdf = gdf.drop(columns='wkt_geom')

# subset of interest
monad = gdf[gdf['Region'] == 'Monadhliath']

wms = ('http://www.getmapping.com/wms/scotland/'
       'scotland.wmsx?login=jhilogin02&password=j54apd')

layer = 'ScotlandBest250mm'

# somewhere for the tiles to go
outdir = ('tiles')

os.mkdir(outdir)

batch_wms_download(monad, wms, layer, outdir, attribute='id',
                       espg='27700')

vrt = os.path.join(outdir, os.path.join(outdir, 'monad.vrt'))

infiles = glob(os.path.join(outdir, '*.tif'))
infiles.sort()

write_vrt(infiles, vrt)

acdir = os.path.join(maindir, 'actiles')
os.mkdir(acdir)
# haha ugly
ootlist = [os.path.join(acdir, os.path.split(i)[1])[:-4]+'_ac.shp' for
           i in infiles]

# I ascertained this by looking at each one (usual algos not reliable for this)
#e.g.- far from perfect but quicker than polygon clipping
img = raster2array(infiles[0])
# this doesn't return anything just to mess about with threshold
image_thresh(img)

# having "established" what they are for each image
threshes = [70, 55, 50, 55, 55, 55]

# could be parallel....
# Using the whole image (thus energy balance is image wide - see further down 
# per blob alternative)
for f, i, t in tqdm(zip(infiles, ootlist, threshes)):
    
    do_ac(f, i, iterations=10, thresh=t,
          smoothing=1, lambda1=1, lambda2=1, area_thresh=4, vis=True)

"""
Alternatively you could draw them roughly then use ac to alter them per polygon
 though care must be taken that the polygon or respective neigbourhood around it
 (ie bufs 1 & 2) doesn't fall of edge of image (see args in lib or src)
 If there are overlapping polygons beforehand best disolve 
 then single > multipart them 
"""
inshp = ('my/erosion.shp')
outshp = ('my/acedit.shp')
ms_snake(inshp, vrt, outshp,  band=2, buf1=0, buf2=0, 
           algo="ACWE", nodata_value=0,
           iterations=50,  smoothing=1, lambda1=1, lambda2=1,
           threshold='auto', balloon=-1)














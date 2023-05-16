#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  2 16:24:26 2023

Functions taken from my own lib (https://github.com/Ciaran1981/geospatial-learn)
for the sake of convenience should you not wish to install it

@author: Ciaran Robb
"""
from osgeo import gdal, ogr, osr
import numpy as np
import geopandas as gpd
import os
from owslib.wms import WebMapService
from io import BytesIO
from joblib import Parallel, delayed
import napari
import dask.array as da
from skimage.transform import rescale
from skimage import color, exposure
import morphsnakes as ms
import sys
import matplotlib.pyplot as plt
import scipy.ndimage as nd
from tqdm import tqdm

gdal.UseExceptions()
ogr.UseExceptions()
osr.UseExceptions()

def do_ac(inras, outshp, iterations=10, thresh=75,
          smoothing=1, lambda1=1, lambda2=1, area_thresh=4, vis=True,
          chess=None):
    """
    Given an image (rgb2gray'd in function here), initialise active contours
    based on either an image threshold value(recommended) or chessboard pattern.
    Result saved to polygon shapefile
    
    Parameters
    ----------
    
    inras: string
            input raster
    
    outshp: string
            output shapefile
    
    iterations: uint
        Number of iterations to run. Stabalises rapidly so not many required
        
    thresh: int
            the image threshold (uint8) required
        
    smoothing : uint, optional
    
        Number of times the smoothing operator is applied per iteration.
        Reasonable values are around 1-4. Larger values lead to smoother
        segmentations.
    
    lambda1: float, optional
    
        Weight parameter for the outer region. If `lambda1` is larger than
        `lambda2`, the outer region will contain a larger range of values than
        the inner region.
        
    lambda2: float, optional
    
        Weight parameter for the inner region. If `lambda2` is larger than
        `lambda1`, the inner region will contain a larger range of values than
        the outer region.
    
    threshold: float, optional
    
        Areas of the image with a value smaller than this threshold will be
        considered borders. The evolution of the contour will stop in this
        areas.
        
    balloon: float, optional
    
        Balloon force to guide the contour in non-informative areas of the
        image, i.e., areas where the gradient of the image is too small to push
        the contour towards a border. A negative value will shrink the contour,
        while a positive value will expand the contour in these areas. Setting
        this to zero will disable the balloon force.
    """
    
    rgb=raster2array(inras, bands=[1,2,3])

    img = color.rgb2gray(rgb)

    img = exposure.rescale_intensity(img, out_range='uint8')

    #ut.image_thresh(img)
    
    if chess is not None:
        bw = ms.checkerboard_level_set((img.shape), chess)
    else:
        bw = img < thresh
    
    if vis == True:
        callback = visual_callback_2d(img)
    
    ac = ms.morphological_chan_vese(img, iterations=iterations,
                       init_level_set=bw,
                       smoothing=smoothing, lambda1=lambda1,
                       lambda2=lambda2, iter_callback=callback)

    ootac = outshp[:-3]+'tif'
    array2raster(ac, 1, inras, ootac, dtype=1)
    
    polygonize(ootac, outshp, 'DN')
    gdf = gpd.read_file(outshp)
    gdf["Area"] = gdf['geometry'].area
    gdf = gdf[gdf.Area > area_thresh]
    gdf.to_file(outshp)

def image_thresh(image):
    
    """
    A Napari-based viewer to threshold an image
    
    Parameters
    ----------
    
    image: nparray
    
    """

    image = exposure.rescale_intensity(image, out_range='uint8')
    
    if image.shape[0] > 4000:
        image = rescale(image, 0.5, preserve_range=True, anti_aliasing=True)
        image = np.uint8(image)
    
    def threshold(image, t):
        arr = da.from_array(image, chunks=image.shape)
        return arr > t
    
    all_thresholds = da.stack([threshold(image, t) for t in np.arange(255)])
    
    viewer = napari.view_image(image, name='input image')
    viewer.add_image(all_thresholds,
        name='thresholded', colormap='magenta', blending='additive'
    )


def batch_wms_download(gdf, wms, layer, outdir, attribute='id',
                       espg='27700', res=0.25):
    
    """
    Download a load of wms tiles with georeferencing
    
    Parameters
    ----------
    
    gdf: geopandas gdf
    
    wms: string 
        the wms addresss
    
    layer: string
        the wms layer
    
    espg: string
            the proj espg
    
    outfile: string
              path to outfile
    
    res: int
            per pixel resolution of imagery in metres
    
    """

    
    rng = np.arange(0, gdf.shape[0])
    
    # assuming each tile is the same size
    bbox = gdf.bounds.iloc[0].tolist()
    # for the img_size
    div = int(1 / res) # must be an int otherwise wms doesn't accept
    # in case it is not a fixed tile size for our aoi
    img_size = (int(bbox[2]-bbox[0])*div,  int(bbox[3]-bbox[1])*div)
    
    outfiles = [os.path.join(outdir, a+'.tif') for a in gdf.id.to_list()]
    
    _ = Parallel(n_jobs=gdf.shape[0],
             verbose=2)(delayed(wmsGrabber)(gdf.bounds.iloc[i].tolist(),
                        img_size, wms, layer,
                        outfiles[i], espg=espg, res=res) for i in rng)

def wmsGrabber(bbox, image_size, wms, layer, outfile, espg='27700', 
               res=0.25):
    
    # Cheers Fraser, with some mods
    
    """
    Return a wms tile from a given source and optionally write to disk with 
    georef
    
    Parameters
    ----------
    
    bbox: list or tuple
            xmin, ymin, xmax, ymax
    
    image_size: tuple
                image x,y dims 
    
    wms: string 
        the wms addresss
        
    layer: string 
        the wms (sub)layer    
    
    espg: string
            the proj espg
    
    outfile: string
              path to outfile, if None only array is returned
    
    """
    
    wms = WebMapService(wms, version='1.1.1')
    
    wms_img = wms.getmap(layers=[layer],
                        srs='EPSG:'+espg,
                        bbox=(bbox[0], bbox[1], bbox[2], bbox[3]),
                        size=image_size,
                        format='image/png',
                        transparent=True
                        )

    f_io = BytesIO(wms_img.read())
    img = plt.imread(f_io)
    
    np2gdal = {"uint8": 1,"int8": 1,"uint16": 2,"int16": 3,
               "uint32": 4,"int32": 5, "float32": 6, 
               "float64": 7, "complex64": 10, "complex128": 11}
    
    
    if outfile != None:
        
        dtpe = np2gdal[str(img.dtype)]
        
        bbox2raster(img, img.shape[2], bbox, outfile, pixel_size=res,  
                    proj=int(espg), dtype=dtpe, FMT='Gtiff')
    
    return img

def bbox2raster(array, bands, bbox, outras, pixel_size=0.25,  proj=27700,
                dtype=5, FMT='Gtiff'):
    
    """
    Using a bounding box and other information georef an image and write to disk
    
    Parameters
    ----------      
    array: np array
            a numpy array.
    
    bands: int
            the no of bands.
    
    bbox: list or tuple
        xmin, ymin, xmax, ymax
    
    pixel_size: int
                pixel size in metres (unless proj is degrees!)
    
    outras: string
             the path of the output raster.
    
    proj: int
         the espg code eg 27700 for osgb
    
    dtype: int 
            though you need to know what the number represents!
            a GDAL datatype (see the GDAL website) e.g gdal.GDT_Int32 = 5
    
    FMT: string 
           (optional) a GDAL raster format (see the GDAL website) eg Gtiff, KEA.
    
    """
    # dimensions & ref coords
    x_pixels = array.shape[1]
    y_pixels = array.shape[0] 
    
    x_min = bbox[0]
    y_max = bbox[3]
    
    driver = gdal.GetDriverByName(FMT)
    
    # Set params for output raster
    ds = driver.Create(
         outras, 
         x_pixels,
         y_pixels,
         bands,
         dtype)

    ds.SetGeoTransform((
        x_min,        # rgt[0]
        pixel_size,   # rgt[1]
        0,            # rgt[2]
        y_max,        # rgt[3]
        0,            # rgt[4]
        -pixel_size)) # rgt[5]
    
    # georef
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(proj)    
    ds.SetProjection(srs.ExportToWkt())
    # Write 
    if bands == 1:
        ds.GetRasterBand(1).WriteArray(array)

    else:
    # Loop through bands - not aware of quicker way when writing
        for band in range(1, bands+1):
            ds.GetRasterBand(band).WriteArray(array[:, :, band-1])
    # Flush to disk
    ds.FlushCache()  
    ds=None

def array2raster(array, bands, inRaster, outRas, dtype, FMT=None):
    
    """
    Save a raster from a numpy array using the geoinfo from another.
    
    Parameters
    ----------      
    array: np array
            a numpy array.
    
    bands: int
            the no of bands. 
    
    inRaster: string
               the path of a raster.
    
    outRas: string
             the path of the output raster.
    
    dtype: int 
            though you need to know what the number represents!
            a GDAL datatype (see the GDAL website) e.g gdal.GDT_Int32
    
    FMT: string 
           (optional) a GDAL raster format (see the GDAL website) eg Gtiff, HFA, KEA.
        
    
    """

    if FMT == None:
        FMT = 'Gtiff'
    
    inras = gdal.Open(inRaster, gdal.GA_ReadOnly)    
    
    x_pixels = inras.RasterXSize  
    y_pixels = inras.RasterYSize  
    geotransform = inras.GetGeoTransform()
    PIXEL_SIZE = geotransform[1]  
    x_min = geotransform[0]
    y_max = geotransform[3]
    # x_min & y_max are like the "top left" corner.
    projection = inras.GetProjection()
    geotransform = inras.GetGeoTransform()   

    driver = gdal.GetDriverByName(FMT)

    dataset = driver.Create(
        outRas, 
        x_pixels,
        y_pixels,
        bands,
        dtype)

    dataset.SetGeoTransform((
        x_min,    # 0
        PIXEL_SIZE,  # 1
        0,                      # 2
        y_max,    # 3
        0,                      # 4
        -PIXEL_SIZE))    

    dataset.SetProjection(projection)
    if bands == 1:
        dataset.GetRasterBand(1).WriteArray(array)
        dataset.FlushCache()  # Write to disk.
        dataset=None
        #print('Raster written to disk')
    else:
    # Here we loop through bands
        for band in range(1,bands+1):
            Arr = array[:,:,band-1]
            dataset.GetRasterBand(band).WriteArray(Arr)
        dataset.FlushCache()  # Write to disk.
        dataset=None
        #print('Raster written to disk')
        
def raster2array(inRas, bands=[1]):
    
    """
    Read a raster and return an array, either single or multiband

    
    Parameters
    ----------
    
    inRas: string
                  input  raster 
                  
    bands: list
                  a list of bands to return in the array
    
    """
    rds = gdal.Open(inRas)
   
   
    if len(bands) ==1:
        # then we needn't bother with all the crap below
        inArray = rds.GetRasterBand(bands[0]).ReadAsArray()
        
    else:
        #   The nump and gdal dtype (ints)
        #   {"uint8": 1,"int8": 1,"uint16": 2,"int16": 3,"uint32": 4,"int32": 5,
        #    "float32": 6, "float64": 7, "complex64": 10, "complex128": 11}
        
        # a numpy gdal conversion dict - this seems a bit long-winded
        dtypes = {"1": np.uint8, "2": np.uint16,
              "3": np.int16, "4": np.uint32,"5": np.int32,
              "6": np.float32,"7": np.float64,"10": np.complex64,
              "11": np.complex128}
        rdsDtype = rds.GetRasterBand(1).DataType
        inDt = dtypes[str(rdsDtype)]
        
        inArray = np.zeros((rds.RasterYSize, rds.RasterXSize, len(bands)), dtype=inDt) 
        for idx, band in enumerate(bands):  
            rA = rds.GetRasterBand(band).ReadAsArray()
            inArray[:, :, idx]=rA
   
   
    return inArray

def write_vrt(infiles, outfile):
    
    """
    Parameters
    ----------
    
    infiles: list
            list of files
    
    outfile: string
                the output .vrt

    """
    
    
    virtpath = outfile
    outvirt = gdal.BuildVRT(virtpath, infiles)
    outvirt.FlushCache()
    outvirt=None

def polygonize(inRas, outPoly, outField=None,  mask = True, band = 1, 
               filetype="ESRI Shapefile"):
    
    """ 
    Polygonise a raster

    Parameters
    -----------   
      
    inRas: string
            the input image   
        
    outPoly: string
              the output polygon file path 
        
    outField: string (optional)
             the name of the field containing burnded values

    mask: bool (optional)
            use the input raster as a mask

    band: int
           the input raster band
            
    """    
    
    #TODO speed this up   

    options = []
    src_ds = gdal.Open(inRas)
    if src_ds is None:
        print('Unable to open %s' % inRas)
        sys.exit(1)
    
    try:
        srcband = src_ds.GetRasterBand(band)
    except RuntimeError as e:
        # for example, try GetRasterBand(10)
        print('Band ( %i ) not found')
        print(e)
        sys.exit(1)
    if mask == True:
        maskband = src_ds.GetRasterBand(band)
        options.append('-mask')
    else:
        mask = False
        maskband = None
    
#    srs = osr.SpatialReference()
#    srs.ImportFromWkt( src_ds.GetProjectionRef() )
    
    ref = src_ds.GetSpatialRef()
    dst_layername = outPoly
    drv = ogr.GetDriverByName(filetype)
    dst_ds = drv.CreateDataSource( dst_layername)
    dst_layer = dst_ds.CreateLayer(dst_layername, srs=ref)
    
    if outField is None:
        dst_fieldname = 'DN'
        fd = ogr.FieldDefn( dst_fieldname, ogr.OFTInteger)
        dst_layer.CreateField( fd )
        dst_field = dst_layer.GetLayerDefn().GetFieldIndex(dst_fieldname)

    
    else: 
        dst_field = dst_layer.GetLayerDefn().GetFieldIndex(outField)

    gdal.Polygonize(srcband, maskband, dst_layer, dst_field,
                    callback=gdal.TermProgress)
    dst_ds.FlushCache()
    
    srcband = None
    src_ds = None
    dst_ds = None

def ms_snake(inShp, inRas, outShp,  band=2, buf1=0, buf2=0, algo="ACWE", nodata_value=0,
          iterations=200,  smoothing=1, lambda1=1, lambda2=1, threshold='auto', 
          balloon=-1):
    
    """ 
    Deform a polygon using active contours on the values of an underlying raster.
    
    This uses morphsnakes and explanations are from there.
    
    Parameters
    ----------
    
    inShp: string
                  input shapefile
        
    inRas: string
                  input raster
    outShp: string
                  output shapefile
        
    band: int
           an integer val eg - 2

    algo: string
           either "GAC" (geodesic active contours) or the default "ACWE" (active contours without edges)
    buf1: int
           the buffer if any in map units for the bounding box of the poly which
           extracts underlying pixel values.
           
    buf2: int
           the buffer if any in map units for the expansion or contraction
           of the poly which will initialise the active contour. 
           This is here as you may wish to adjust the init polygon so it does not
           converge on a adjacent one or undesired area. 
          
    nodata_value: numerical
                   If used the no data val of the raster

    iterations: uint
        Number of iterations to run.
        
    smoothing : uint, optional
    
        Number of times the smoothing operator is applied per iteration.
        Reasonable values are around 1-4. Larger values lead to smoother
        segmentations.
    
    lambda1: float, optional
    
        Weight parameter for the outer region. If `lambda1` is larger than
        `lambda2`, the outer region will contain a larger range of values than
        the inner region.
        
    lambda2: float, optional
    
        Weight parameter for the inner region. If `lambda2` is larger than
        `lambda1`, the inner region will contain a larger range of values than
        the outer region.
    
    threshold: float, optional
    
        Areas of the image with a value smaller than this threshold will be
        considered borders. The evolution of the contour will stop in this
        areas.
        
    balloon: float, optional
    
        Balloon force to guide the contour in non-informative areas of the
        image, i.e., areas where the gradient of the image is too small to push
        the contour towards a border. A negative value will shrink the contour,
        while a positive value will expand the contour in these areas. Setting
        this to zero will disable the balloon force.
        
    """    
    
    # Partly inspired by the Heikpe paper...
    # TODO read rgb/3band in and convert 2 gray
    
    rds = gdal.Open(inRas, gdal.GA_ReadOnly)
    #assert(rds)
    rb = rds.GetRasterBand(band)
    rgt = rds.GetGeoTransform()
    
    
    if nodata_value:
        nodata_value = float(nodata_value)
        rb.SetNoDataValue(nodata_value)

    vds = ogr.Open(inShp, 1)  # TODO maybe open update if we want to write stats

    vlyr = vds.GetLayer(0)


    mem_drv = ogr.GetDriverByName('Memory')
    driver = gdal.GetDriverByName('MEM')

    # Loop through vectors
    features = np.arange(vlyr.GetFeatureCount())

    
    outDataset = _copy_dataset_config(rds, outMap = outShp[:-4]+'.tif',
                                     bands = 1, )
    
    outBnd = outDataset.GetRasterBand(1)
    

    for label in tqdm(features):

        feat = vlyr.GetFeature(label)
#        if feat is None:
#            continue
        geom = feat.geometry()
        buff = geom.Buffer(buf1)
        
        src_offset = _bbox_to_pixel_offsets(rgt, buff)
        
        src_offset = list(src_offset)
        
        for idx, off in enumerate(src_offset):
            if off <=0:
                src_offset[idx]=0
               
        src_array = rb.ReadAsArray(src_offset[0], src_offset[1], src_offset[2],
                               src_offset[3])
        if src_array is None:
            src_array = rb.ReadAsArray(src_offset[0]-1, src_offset[1], src_offset[2],
                               src_offset[3])
            if src_array is None:
#                rejects.append(feat.GetFID())
                continue

        # calculate new geotransform of the feature subset
        new_gt = (
        (rgt[0] + (src_offset[0] * rgt[1])),
        rgt[1],
        0.0,
        (rgt[3] + (src_offset[1] * rgt[5])),
        0.0,
        rgt[5])
                    
        # Create a temporary vector layer in memory
        mem_ds = mem_drv.CreateDataSource('out')
        mem_layer = mem_ds.CreateLayer('poly', None, ogr.wkbPolygon)
        mem_layer.CreateFeature(feat.Clone())
#
#        # Rasterize it
        
        rvds = driver.Create('', src_offset[2], src_offset[3], 1, gdal.GDT_Byte)
        rvds.SetGeoTransform(new_gt)
        gdal.RasterizeLayer(rvds, [1], mem_layer, burn_values=[1])
        rv_array = rvds.ReadAsArray()
        
        if buf2 < 0:
            dist = nd.morphology.distance_transform_edt(rv_array)
        else:
            dist = nd.morphology.distance_transform_edt(np.logical_not(rv_array))
        
        # covert the dist raster to the actual units
        dist *= rgt[1]
        
        # expand or contract the blob
        if buf2 != 0:
            if buf2 > 0:                
                rv_array = dist <=buf2
            else:
                rv_array = dist >=abs(buf2)                      
        
        if algo == "ACWE":       
        
            bw = ms.morphological_chan_vese(src_array, iterations=iterations,
                                   init_level_set=rv_array,
                                   smoothing=smoothing, lambda1=lambda1,
                                   lambda2=lambda2)
        if algo == "GAC":
            gimg = ms.inverse_gaussian_gradient(src_array)
            bw = ms.morphological_geodesic_active_contour(gimg, iterations, rv_array,
                                             smoothing=smoothing, threshold=threshold,
                                             balloon=balloon)

        
        segoot = np.int32(bw)
        segoot*=int(label)+1
        
        # very important not to overwrite results
        if label > 0:
            ootArray = outBnd.ReadAsArray(src_offset[0], src_offset[1], src_offset[2],
                               src_offset[3])
            ootArray[segoot==label+1]=label+1
            outBnd.WriteArray(ootArray, src_offset[0], src_offset[1])
        else:
    
            outBnd.WriteArray(segoot, src_offset[0], src_offset[1])
        
        del segoot, bw
        feat = vlyr.GetNextFeature()
        
    
    outDataset.FlushCache()
    
    outDataset=None
    vds = None
    
    # This is a hacky solution for now really, but it works well enough!
    polygonize(outShp[:-4]+'.tif', outShp, outField='id',  mask = True, band = 1)

def _copy_dataset_config(inDataset, FMT = 'Gtiff', outMap = 'copy',
                         dtype = gdal.GDT_Int32, bands = 1):
    """Copies a dataset without the associated rasters.

    """

    
    x_pixels = inDataset.RasterXSize  # number of pixels in x
    y_pixels = inDataset.RasterYSize  # number of pixels in y
    geotransform = inDataset.GetGeoTransform()
    PIXEL_SIZE = geotransform[1]  # size of the pixel...they are square so thats ok.
    #if not would need w x h
    x_min = geotransform[0]
    y_max = geotransform[3]
    # x_min & y_max are like the "top left" corner.
    projection = inDataset.GetProjection()
    geotransform = inDataset.GetGeoTransform()   
    #dtype=gdal.GDT_Int32
    driver = gdal.GetDriverByName(FMT)
    
    # Set params for output raster
    outDataset = driver.Create(
        outMap, 
        x_pixels,
        y_pixels,
        bands,
        dtype)

    outDataset.SetGeoTransform((
        x_min,    # 0
        PIXEL_SIZE,  # 1
        0,                      # 2
        y_max,    # 3
        0,                      # 4
        -PIXEL_SIZE))
        
    outDataset.SetProjection(projection)
    
    return outDataset

def _bbox_to_pixel_offsets(rgt, geom):
    
    """ 
    Internal function to get pixel geo-locations of bbox of a polygon
    
    Parameters
    ----------
    
    rgt: array
         raster geotransform
          
    geom: shapely.geometry
           Structure defining geometry
    
    Returns
    -------
    xoffset: int
           
    yoffset: iny
           
    xcount: int
             rows of bounding box
             
    ycount: int
             columns of bounding box
    """
    
    xOrigin = rgt[0]
    yOrigin = rgt[3]
    pixelWidth = rgt[1]
    pixelHeight = rgt[5]
    ring = geom.GetGeometryRef(0)
    numpoints = ring.GetPointCount()
    pointsX = []; pointsY = []
    
    if (geom.GetGeometryName() == 'MULTIPOLYGON'):
        count = 0
        pointsX = []; pointsY = []
        for polygon in geom:
            geomInner = geom.GetGeometryRef(count)
            ring = geomInner.GetGeometryRef(0)
            numpoints = ring.GetPointCount()
            for p in range(numpoints):
                    lon, lat, z = ring.GetPoint(p)
                    pointsX.append(lon)
                    pointsY.append(lat)
            count += 1
    elif (geom.GetGeometryName() == 'POLYGON'):
        ring = geom.GetGeometryRef(0)
        numpoints = ring.GetPointCount()
        pointsX = []; pointsY = []
        for p in range(numpoints):
                lon, lat, z = ring.GetPoint(p)
                pointsX.append(lon)
                pointsY.append(lat)
            
    xmin = min(pointsX)
    xmax = max(pointsX)
    ymin = min(pointsY)
    ymax = max(pointsY)

    # Specify offset and rows and columns to read
    xoff = int((xmin - xOrigin)/pixelWidth)
    yoff = int((yOrigin - ymax)/pixelWidth)
    xcount = int((xmax - xmin)/pixelWidth)+1
    ycount = int((ymax - ymin)/pixelWidth)+1

    return (xoff, yoff, xcount, ycount)


def visual_callback_2d(background, fig=None):
    
    # From morphsnakes
    
    """
    Returns a callback than can be passed as the argument `iter_callback`
    of `morphological_geodesic_active_contour` and
    `morphological_chan_vese` for visualizing the evolution
    of the levelsets. Only works for 2D images.
    Parameters
    ----------
    background : (M, N) array
        Image to be plotted as the background of the visual evolution.
    fig : matplotlib.figure.Figure
        Figure where results will be drawn. If not given, a new figure
        will be created.
    Returns
    -------
    callback : Python function
        A function that receives a levelset and updates the current plot
        accordingly. This can be passed as the `iter_callback` argument of
        `morphological_geodesic_active_contour` and
        `morphological_chan_vese`.
    """

    # Prepare the visual environment.
    if fig is None:
        fig = plt.figure()
    fig.clf()
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.imshow(background, cmap=plt.cm.gray)

    ax2 = fig.add_subplot(1, 2, 2)
    ax_u = ax2.imshow(np.zeros_like(background), vmin=0, vmax=1)
    plt.pause(0.001)

    def callback(levelset):

        if ax1.collections:
            del ax1.collections[0]
        ax1.contour(levelset, [0.5], colors='r')
        ax_u.set_data(levelset)
        fig.canvas.draw()
        plt.pause(0.001)

    return callback

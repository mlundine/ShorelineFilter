"""
Author: Mark Lundine, USGS

Using spatial kernel density estimate (a heatmap) to construct a shoreline-change
envelope based upon extracted shoreline points from CoastSeg.
Output can be used as a shoreline_extraction_area in a follow-up run of CoastSeg models.
Might not need reference shorelines anymore.
Might be able to use output to draw better transects.

Points to KDE map to otsu threshold map to shoreline-change envelope.
"""
## Imports
import geopandas as gpd
import pandas as pd
import os
import glob
import numpy as np
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import box
from skimage.filters import threshold_otsu
from skimage.filters import threshold_multiotsu
from scipy.ndimage import gaussian_filter

def get_script_path():
    return os.path.dirname(os.path.abspath(__file__))

def get_immediate_subdirectories(a_dir):
    return [name for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))]

def join_model_scores(good_bad, good_bad_seg, shorelines_points, img_type):
    shorelines_points_gdf = gpd.read_file(shorelines_points)
    shorelines_points_gdf['date'] = pd.to_datetime(shorelines_points_gdf['date'], utc=True)

    try:
        cols = list(shorelines_points_gdf.columns)
        keep_cols = ['date', 'satname', 'geoaccuracy', 'cloud_cover', 'geometry']
        for col in cols:
            if col not in keep_cols:
                shorelines_points_gdf = shorelines_points_gdf.drop(columns=[col])
    except:
        pass

    good_bad_df = pd.read_csv(good_bad)
    good_bad_seg_df = pd.read_csv(good_bad_seg)

    dts = [None]*len(good_bad_df)
    for i in range(len(good_bad_df)):
        dt = os.path.basename(good_bad_df['im_paths'].iloc[i])
        idx = dt.find('_RGB')
        dt = dt[0:idx]
        dts[i] = dt
    good_bad_df['dates'] = dts
    good_bad_df['dates'] = pd.to_datetime(good_bad_df['dates'], utc=True,
                                          format='%Y-%m-%d-%H-%M-%S')

    dts_seg = [None]*len(good_bad_seg_df)
    for i in range(len(good_bad_seg_df)):
        dt = os.path.basename(good_bad_seg_df['im_paths'].iloc[i])
        idx = dt.find('_'+img_type)
        dt = dt[0:idx]
        dts_seg[i] = dt
    good_bad_seg_df['dates'] = dts_seg
    good_bad_seg_df['dates'] = pd.to_datetime(good_bad_seg_df['dates'], utc=True,
                                          format='%Y-%m-%d-%H-%M-%S')

    shorelines_points_gdf = shorelines_points_gdf.merge(good_bad_df,
                                               left_on='date',
                                               right_on='dates',
                                               suffixes=['', '_image']
                                                        )
    shorelines_points_gdf = shorelines_points_gdf.merge(good_bad_seg_df,
                                               left_on='date',
                                               right_on='dates',
                                               suffixes=['', '_seg']
                                                        )
    shorelines_points_gdf = shorelines_points_gdf.drop(columns=['Unnamed: 0_seg', 
                                                                'Unnamed: 0',
                                                                'dates_seg',
                                                                ]
                                                                )
    shorelines_points_gdf.to_file(shorelines_points)

def wgs84_to_utm_df(geo_df):
    """
    Converts wgs84 to UTM
    inputs:
    geo_df (geopandas dataframe): a geopandas dataframe in wgs84
    outputs:
    geo_df_utm (geopandas  dataframe): a geopandas dataframe in utm
    """
    utm_crs = geo_df.estimate_utm_crs()
    gdf_utm = geo_df.to_crs(utm_crs)
    return gdf_utm

def utm_to_wgs84_df(geo_df):
    """
    Converts utm to wgs84
    inputs:
    geo_df (geopandas dataframe): a geopandas dataframe in utm
    outputs:
    geo_df_wgs84 (geopandas  dataframe): a geopandas dataframe in wgs84
    """
    wgs84_crs = 'epsg:4326'
    gdf_wgs84 = geo_df.to_crs(wgs84_crs)
    return gdf_wgs84

def min_max_normalize(arr):
    min_val = np.min(arr)
    max_val = np.max(arr)
    return (arr - min_val) / (max_val - min_val)

def point_density_grid(points_path, save_path, cell_size):
    """
    This is a slow way of making a point density grid.
    Not used but keeping in here in case this is useful.
    inputs:
    points_path (str): path to geojson of points
    cell_size (int): cell size of output density grid, this will be in units of the input coordinate system
    outputs:
    point_density_grid_path (str): path to the density grid as a geotiff
    """
    points = gpd.read_file(points_path)
    points_utm = wgs84_to_utm_df(points)
    points_exploded = points_utm.explode(index_parts=True)

    # Create a grid
    minx, miny, maxx, maxy = points_exploded.total_bounds
    x_coords = np.arange(minx, maxx + cell_size, cell_size)
    y_coords = np.arange(miny, maxy + cell_size, cell_size)
    grid_cells = [box(x, y, x + cell_size, y + cell_size) for x in x_coords for y in y_coords]
    grid = gpd.GeoDataFrame({'geometry': grid_cells})

    # Count points in each grid cell
    grid['point_count'] = grid.apply(lambda cell: points_exploded.within(cell.geometry).sum(), axis=1)

    # Create the raster
    transform = from_origin(minx, maxy, cell_size, cell_size)
    raster = np.zeros((len(y_coords), len(x_coords)))
    for idx, row in grid.iterrows():
        row_idx = int((maxy - row.geometry.bounds[1]) / cell_size)
        col_idx = int((row.geometry.bounds[0] - minx) / cell_size)
        raster[row_idx, col_idx] = row['point_count']

    raster = min_max_normalize(raster)
    print(max(raster.ravel()))
    print(min(raster.ravel()))
    raster = gaussian_filter(raster, sigma=3)  # Adjust sigma for smoothing level  
    raster = min_max_normalize(raster)
  
    print(max(raster.ravel()))
    print(min(raster.ravel()))
    ##making a path for the output grid
    point_density_grid_path = save_path
    ##save the grid
    with rasterio.open(
        point_density_grid_path, 'w', driver='GTiff', height=raster.shape[0],
        width=raster.shape[1], count=1, dtype='float32',
        crs=points_utm.crs, transform=transform
    ) as dst:
        dst.write(raster, 1)

    return point_density_grid_path

def compute_otsu_threshold(in_tiff, out_tiff):
    """
    Otsu binary thresholding on a geotiff.
    inputs:
    in_tiff (str): path to the input geotiff
    out_tiff (str): path to the output geotiff
    outputs:
    out_tiff (str): path to the output geotiff
    """
    with rasterio.open(in_tiff) as src:
        image = src.read(1)  

    # Compute Otsu's threshold
    # Need to make nodata values zero or else the threshold will be just data vs. nodata
    # This works for our example because point density is always greater than or equal to zero.
    image[image==src.meta['nodata']]=0
    print(min(image.ravel()))
    print(max(image.ravel()))
    threshold = threshold_otsu(image)
    thresholds = threshold_multiotsu(image)

    # Apply the threshold to create a binary image
    binary_image = image > min(thresholds)

    # Define the metadata for the new geotiff
    transform = from_origin(src.bounds.left, src.bounds.top, src.res[0], src.res[1])
    new_meta = src.meta.copy()
    new_meta.update({
        'dtype': 'uint16',
        'count': 1,
        'transform': transform,
        'nodata':None
    })

    # Save the binary image
    with rasterio.open(out_tiff, 'w', **new_meta) as dst:
        dst.write(binary_image.astype(np.uint8), 1)
    
    return out_tiff

def binary_raster_to_vector(in_tiff, out_geojson):
    """
    Converts a binary raster to a vector file using gdal_polygonize.
    Uses itself as a mask to isolate the raster where cell values == 1.
    Currently running gdal_polygonize as a script from os.system, might want to change this
    inputs:
    in_tiff (str): path to the input binary raster as geotiff
    out_geojson (str): path to the output vector file as geojson
    outputs:
    out_geojson (str): path to the output vector file as geojson
    """
    module = os.path.join(get_script_path(), 'gdal_polygonize.py')
    cmd = 'python ' + module + ' ' + in_tiff + ' ' + out_geojson + ' -mask ' + in_tiff + ' -ogr_format GeoJSON'
    os.system(cmd)
    return out_geojson

def buffer_otsu_vector(in_geojson, out_geojson, buffer_value):
    """
    Just adding a buffer radius to the vector file. 
    The Otsu thresholding on point density can be a bit tight (especially in the seaward direction) for the shoreline envelope.
    This is likely due to the temporal bias towards the present in a lot of cases.
    Should add a bit of buffer to make the envelope a little more forgiving.
    inputs:
    in_geojson (str): the shoreline envelope
    out_geojson (str): path to the output shoreline envelope with additional buffer
    buffer_value (int): buffer radius in meters
    outputs:
    out_geojson (str): path to the output shoreline envelope with additional buffer
    """
    ## Do the buffering
    otsu_vector = gpd.read_file(in_geojson)
    buffer = otsu_vector.buffer(buffer_value)

    ## Convert back to WGS84
    buffer_wgs84 = utm_to_wgs84_df(buffer)
    buffer_wgs84_feat = buffer_wgs84.union_all()
    buffer_wgs84_final = gpd.GeoDataFrame({'id':[0],
                                      'geometry':[buffer_wgs84_feat]},crs=buffer_wgs84.crs)
    buffer_wgs84_final.to_file(out_geojson)
    return out_geojson

def get_point_density_kde(extracted_shorelines_points_path, 
                          point_density_kde_path,
                          otsu_path,
                          shoreline_change_envelope_path,
                          shoreline_change_envelope_buffer_path,
                          kde_radius=80,
                          cell_size=20,
                          buffer=50):
    """
    Makes a point density heat map and saves as a geotiff using spatial-kde
    inputs:
    extracted_shorelines_points_path (str): path to the extracted shorelines as points
    point_density_kde_path (str): path to save the result to
    kde_radius (int, optional): radius for the spatial KDE, making this smaller makes the heatmap finer, default is 80 meters
    cell_size (int, optional): resolution of the output heatmap in meters, default is 15 m, can go finer if needed but will slow this down
    outputs:
    point_density_kde_path (str): path to the result
    """
    points = gpd.read_file(extracted_shorelines_points_path)

    ## need in utm
    #points_utm = wgs84_to_utm_df(points)

    ## need each individual point
    #points_exploded = points_utm.explode(index_index=True, index_parts=False)

    ## calling the spatial kde function
    print('computing density grid')
    if os.path.isfile(point_density_kde_path)==False:
        point_density_grid(extracted_shorelines_points_path, point_density_kde_path, cell_size)
    print('computing otsu threshold')
    if os.path.isfile(otsu_path)==False:
        compute_otsu_threshold(point_density_kde_path, otsu_path)
    print('converting otsu to vector')
    if os.path.isfile(shoreline_change_envelope_path)==False:
        binary_raster_to_vector(otsu_path, shoreline_change_envelope_path)
    print('making final vector')
    if os.path.isfile(shoreline_change_envelope_buffer_path)==False:
        buffer_otsu_vector(shoreline_change_envelope_path, shoreline_change_envelope_buffer_path, buffer_value=buffer)

    return shoreline_change_envelope_buffer_path

def get_point_density_kde_multiple_sessions(home,
                                            kde_radius=80,
                                            cell_size=15,
                                            buffer=50,
                                            im_thresh=0.335,
                                            seg_thresh=0.457,
                                            kde_thresh=0.10,
                                            img_type='RGB'):
    """
    Computes spatial kde on multiple coastseg shoreline extraction sessions
    inputs:
    home (str): path to the sessions
    kde_radius (int): radius for the kde
    cell_size (int): cell size of kde raster
    im_thresh (float): threshold for image suitability filter
    seg_thresh (float): threshold for segmentation filter
    kde_thresh (float): threshold for kde filter
    """
    sites = get_immediate_subdirectories(home)
    for site in sites:
        site = os.path.join(home, site)
        print('doing ' + site)
        extracted_shorelines_points_path = os.path.join(site, 'extracted_shorelines_points.geojson')
        good_bad =  os.path.join(site, 'good_bad.csv')
        good_bad_seg = os.path.join(site, 'good_bad_seg.csv')

        join_model_scores(good_bad, good_bad_seg, extracted_shorelines_points_path, img_type)
        
        ##making points files
        ##unfiltered
        points_unfiltered = gpd.read_file(extracted_shorelines_points_path)
        ##just image suitability filter
        points_image_filter = points_unfiltered[points_unfiltered['model_scores']>=im_thresh]
        points_image_filter.to_file(os.path.join(site, 'extracted_shorelines_points_image_filter.geojson'))
        ##just seg filter
        points_seg_filter = points_unfiltered[points_unfiltered['model_scores_seg']>=im_thresh]
        points_seg_filter.to_file(os.path.join(site, 'extracted_shorelines_points_seg_filter.geojson'))
        ##image suitability and seg filter
        points_both_filter = points_image_filter[points_image_filter['model_scores_seg']>=seg_thresh]
        points_both_filter.to_file(os.path.join(site, 'extracted_shorelines_points_image_and_seg_filter.geojson'))
        
        ##make paths for kde files
        point_density_kde_path =  os.path.join(site, 'spatial_kde.tif')
        otsu_path = os.path.join(site, 'spatial_kde_otsu.tif')
        shoreline_change_envelope_path = os.path.join(site, 'shoreline_change_envelope.geojson')
        shoreline_change_envelope_buffer_path = os.path.join(site, 'shoreline_change_envelope_buffer.geojson')

        ##get kde files
        get_point_density_kde(extracted_shorelines_points_path,
                                  point_density_kde_path,
                                  otsu_path,
                                  shoreline_change_envelope_path,
                                  shoreline_change_envelope_buffer_path,
                                  kde_radius=kde_radius,
                                  cell_size=cell_size,
                                  buffer=buffer)
        
        ##need exploded points
        points_both_filter_explode = points_both_filter.explode(index_index=True, index_parts=False)
        points_unfiltered_explode = points_unfiltered.explode(index_index=True, index_parts=False)

        ##need unfiltered in utm
        points_unfiltered_explode = wgs84_to_utm_df(points_unfiltered_explode)
        ##add kde value as field to unfiltered points
        coord_list = [(x, y) for x, y in zip(points_unfiltered_explode["geometry"].x, 
                                             points_unfiltered_explode["geometry"].y)]
        src = rasterio.open(point_density_kde_path)
        points_unfiltered_explode["kde_value"] = [x for x in src.sample(coord_list)]
        for i in range(len(points_unfiltered_explode["kde_value"])):
            points_unfiltered_explode["kde_value"].iloc[i] = float(points_unfiltered_explode["kde_value"].iloc[i][0])
        points_unfiltered_explode["kde_value"] = points_unfiltered_explode["kde_value"].astype(float)

        ##compute overall score
        points_unfiltered_explode["overall_score"] = (points_unfiltered_explode["kde_value"]+points_unfiltered_explode["model_scores"]+points_unfiltered_explode["model_scores_seg"])/3
        points_unfiltered_explode["overall_score"] = min_max_normalize(points_unfiltered_explode["overall_score"])
        
        ##save unfilted points in wgs84, this file now has image suitability, segmentation, and kde scores
        points_unfiltered_explode = utm_to_wgs84_df(points_unfiltered_explode)
        points_unfiltered_explode.to_file(extracted_shorelines_points_path)

        ##make filtered file with kde val
        points_both_filter_explode_kde_vals = points_unfiltered_explode[points_unfiltered_explode['kde_value']>kde_thresh].reset_index(drop=True)
        points_both_filter_explode_kde_vals = points_both_filter_explode_kde_vals[points_both_filter_explode_kde_vals['model_scores']>=im_thresh].reset_index(drop=True)
        points_both_filter_explode_kde_vals = points_both_filter_explode_kde_vals[points_both_filter_explode_kde_vals['model_scores_seg']>=seg_thresh].reset_index(drop=True)
        points_both_filter_explode_kde_vals.to_file(os.path.join(site, 'extracted_shorelines_points_image_and_seg_kde_vals.geojson'))
        
        ##make spatial filtered with kde vals
        envelope = gpd.read_file(shoreline_change_envelope_buffer_path)
        points_image_seg_envelope_filter = gpd.sjoin(points_both_filter_explode_kde_vals, envelope, how="inner", predicate='within')
        points_image_seg_envelope_filter.to_file(os.path.join(site, 'extracted_shorelines_points_image_and_seg_and_kde_filter.geojson'))
            
    return shoreline_change_envelope_buffer_path




            

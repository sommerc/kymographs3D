from kymograph import Kymograph3D, test_rodriguez_rot, convert_and_resample_from_tif, test_square_sampling

def example_zylinder():
    """ Debug example: Show sampled zylinder"""
    test_rodriguez_rot(length=100,radius=2)
    
def example_zylinder_width_phase_shifts():
    """ Debug example: Show sampled zylinder"""
    test_rodriguez_rot(length=100,radius=3, phase_shift=True)
    
def example_square_sampling_points():
    """ Debug example: Show sampled zylinder"""
    test_square_sampling(length=20, width=5)
     
def example_data_preparaton():
    """ Preprocess input multi-page OME tif: """
    tif_file = 'M:/experiments/Experiments_002500/002513/Analysis/2014_03_27(microtubules)/Process_02_3D_tracking_CENPA/cell1_12_crop_3D_Gaussian_ROI_t11_150.tif'
    convert_and_resample_from_tif(tif_file,
                                  'cell1_12_crop.h5',
                                  z_factor=2.35)
def example_1(): 
    """ Example 1: standard kymograph for ids ((0,24), (1,24)), with no extension"""
    kymo1 = Kymograph3D("cell1_12_crop.h5", # input image as constructed by 'convert_and_resample_from_tif'
                        "data_resampled_prefiltered", # default name of resampled image
                        "tracks_pole.txt", # Origin track mate file
                        "tracks_kinetochore.txt", # Destination track mate file
                        [1, 1, 2.35] # Scaling for x, y, z. This has to match the z_factor in 'convert_and_resample_from_tif' 
                        ,)
    kymo1.compute(radius=0, # zylinder radius 
                  aggregation='mean', # mean or max
                  extension=[0,0], # extension in percentage of [origin, destination], if destination equals -1, then the full profile is taken
                  ids=((24,0), (24,1))) # compute kymographs for (origin_id, destination_id)
    kymo1.export(channel_scaling=(2, 0.25)) # scaing of the resulting rgb file
     
def example_2(): 
    """ Example 2: standard and butterfly kymographs for all tracks with pole extension"""
    kymo2 = Kymograph3D("cell1_12_crop.h5", 
                        "data_resampled_prefiltered",
                        "tracks_kinetochore.txt", # Note change of order
                        "tracks_pole.txt",
                        [1, 1, 2.35],)
    kymo2.compute(radius=1, aggregation='max', extension=[0,-1])
     
    kymo2.export_butterfly(channel_scaling=(2, 0.25))
    kymo2.export(channel_scaling=(2, 0.25))

def example_3(): 
    """ Example 3: raw kymographs for all tracks with pole extension"""
    kymo3 = Kymograph3D("cell1_12_crop.h5",
                        "data_resampled_prefiltered",
                        "tracks_pole.txt",
                        "tracks_kinetochore.txt", # Note change of order
                        [1, 1, 2.35],)
    kymo3.compute(radius=1, aggregation='max', extension=[0.1,0.1], ids=((0,84), (1,84)))
    kymo3.export_raw() 
    
def example_4(): 
    """ Example 3: raw kymographs for all tracks without pole extension only on rim"""
    kymo3 = Kymograph3D("cell1_12_crop.h5",
                        "data_resampled_prefiltered",
                        "tracks_pole.txt",
                        "tracks_kinetochore.txt", # Note change of order
                        [1, 1, 2.35],)
    kymo3.compute(radius=3, aggregation='max', extension=[0,0], integration='rim')
    kymo3.export_raw() 
    
    
def example_5(): 
    """ Example 5: standard kymograph for ids ((0,24), (1,24)), with extension, and plane export"""
    kymo1 = Kymograph3D("cell1_12_crop.h5", # input image as generated by 'convert_and_resample_from_tif'
                        "data_resampled_prefiltered", # default name of resampled image
                        "tracks_pole.txt", # Origin track mate file
                        "tracks_kinetochore.txt", # Destination track mate file
                        [1, 1, 2.35] # Scaling for x, y, z. This has to match the z_factor in 'convert_and_resample_from_tif' 
                        ,)
    kymo1.compute(radius=3, # zylinder radius 
                  aggregation='mean', # mean or max
                  extension=[0,-1], # extension in percentage of [origin, destination], if destination equals -1, then the full profile is taken
                  ids=((0,24), (1,25)),
                  export_planes=True,
                  plane_width=31,
                  plane_pixel_width=31
                  ) # compute kymographs for (origin_id, destination_id)


    kymo1.export_planes(channel_scaling=(2,0.25)) # scaing of the resulting rgb file
    kymo1.export(channel_scaling=(2, 0.25)) # scaing of the resulting rgb file
    
def example_6(): 
    """ Example 5: standard kymograph for ids ((0,24), (1,24)), with extension, and plane export"""
    kymo1 = Kymograph3D("test_data.h5", # input image as generated by 'convert_and_resample_from_tif'
                        "data_resampled_prefiltered", # default name of resampled image
                        "test_data_origin.txt", # Origin track mate file
                        "test_data_dest.txt", # Destination track mate file
                        [1, 1, 1] # Scaling for x, y, z. This has to match the z_factor in 'convert_and_resample_from_tif' 
                        ,)
    kymo1.compute(radius=2, # zylinder radius 
                  aggregation='mean', # mean or max
                  extension=[0,-1], # extension in percentage of [origin, destination], if destination equals -1, then the full profile is taken
                  ids=((0,0), (1,0)),
                  export_planes=True,
                  plane_width=11,
                  plane_pixel_width=23
                  ) # compute kymographs for (origin_id, destination_id)


    kymo1.export_planes(channel_scaling=(2,2)) # scaing of the resulting rgb file
    kymo1.export(channel_scaling=(2, 2)) # scaing of the resulting rgb file
    
if __name__ == "__main__":
    # Examples
#     example_zylinder()
#     example_zylinder_width_phase_shifts()
#     example_square_sampling_points()
#     example_4()
#     example_1()
#     example_2()
#     example_3()
    example_5()
    
    print "*"*100
    print "Final feliz"
    
    
    
    
    
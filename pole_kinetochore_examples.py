from kymograph import Kymograph3D, test_rodriguez_rot, convert_and_resample_from_tif

if __name__ == "__main__":
    # Debug example: Show sampled zylinder
    #########################################################################
    test_rodriguez_rot(length=100,radius=1)
    
    
    # Preprocess input multi-page OME tif: 
    #########################################################################
    test_rodriguez_rot(length=100,radius=1)
    if False: # takes some time
        tif_file = 'M:/experiments/Experiments_002500/002513/Analysis/2014_03_27(microtubules)/Process_02_3D_tracking_CENPA/cell1_12_crop_3D_Gaussian_ROI_t11_150.tif'
        convert_and_resample_from_tif(tif_file,
                                      'cell1_12_crop.h5',
                                      z_factor=2.35)
    
    # Example 1: standard kymograph for ids ((0,24), (1,24)), with no extension
    #########################################################################
    kymo1 = Kymograph3D("cell1_12_crop.h5",
                        "data_resampled_prefiltered",
                        "tracks_pole.txt",
                        "tracks_kinetochore.txt",
                        [1, 1, 2.35],)
    kymo1.compute(radius=0, aggregation='mean', extension=[0,0], ids=((0,24), (1,24)))
    kymo1.export(channel_scaling=(2, 0.25))
    
    # Example 2: standard and butterfly kymographs for all tracks with pole extension
    #########################################################################
    kymo2 = Kymograph3D("cell1_12_crop.h5",
                        "data_resampled_prefiltered",
                        "tracks_kinetochore.txt", # Note change of order
                        "tracks_pole.txt",
                        [1, 1, 2.35],)
    kymo2.compute(radius=1, aggregation='max', extension=[0,-1])
    
    kymo2.export_butterfly(channel_scaling=(2, 0.25))
    kymo2.export(channel_scaling=(2, 0.25))
    
    print "*"*100
    print "Final feliz"
    
    
    
    
    
    
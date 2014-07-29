from kymograph import Kymograph3D, test_rodriguez_rot, convert_and_resample_from_tif

def example_zylinder():
    """ Debug example: Show sampled zylinder"""
    test_rodriguez_rot(length=100,radius=1)
     

def example_data_preparaton():
    """ Preprocess input multi-page OME tif: """
    tif_file = 'M:/experiments/Experiments_002500/002513/Analysis/2014_03_27(microtubules)/Process_02_3D_tracking_CENPA/cell1_12_crop_3D_Gaussian_ROI_t11_150.tif'
    convert_and_resample_from_tif(tif_file,
                                  'cell1_12_crop.h5',
                                  z_factor=2.35)
def example_1(): 
    """ Example 1: standard kymograph for ids ((0,24), (1,24)), with no extension"""
    kymo1 = Kymograph3D("cell1_12_crop.h5",
                        "data_resampled_prefiltered",
                        "tracks_pole.txt", # Origin
                        "tracks_kinetochore.txt", # Destination
                        [1, 1, 2.35],)
    kymo1.compute(radius=0, aggregation='mean', extension=[0,0], ids=((0,24), (1,24)))
    kymo1.export(channel_scaling=(2, 0.25))
     
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

    
    
    
if __name__ == "__main__":
    # Examples
    example_zylinder()
    example_1()
    example_2()
    example_3()
    
    print "*"*100
    print "Final feliz"
    
    
    
    
    
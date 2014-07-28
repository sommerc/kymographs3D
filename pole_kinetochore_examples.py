from kymograph import Kymograph3D, test_rodriguez_rot


test_rodriguez_rot(length=100,radius=1)
         
kymo = Kymograph3D("cell1_12_crop.h5",
                            "data_resampled_prefiltered",
#                             "data_resampled",
                            "tracks_kinetochore.txt",
                            "tracks_pole.txt",
                            [1, 1, 2.35],)
kymo.compute(radius=0, aggregation='mean', extension=[0,-1], ids=((0,24), (1,24)))
kymo.compute(radius=0, aggregation='mean', extension=[0,-1])
kymo.export_butterfly(channel_scaling=(2, 0.25))
kymo.export(channel_scaling=(2, 0.25))
import numpy
import vigra

import h5py
from scipy import ndimage
import pandas
import logging
from collections import defaultdict
import os
import pylab
import unittest

logging.basicConfig()
log = logging.getLogger("Kymograph3D")
log.setLevel(logging.DEBUG)

eps = 10e-9

def get_w(x,y,z):
    assert (numpy.linalg.norm([x,y,z]) - 1) < eps
    return numpy.array([[ 0, -z,  y],
                        [ z,  0, -x],
                        [-y,  x,  0]])
    
def get_rodriguies_rotations(x,y,z, num=8, phase_shift=False):
    W = get_w(x,y,z)
    W2 = W.dot(W)
    num = float(num)
    if phase_shift:
        return [numpy.eye(3,3) + numpy.sin(phi) * W + (1-numpy.cos(phi)) * W2 for phi in numpy.linspace(numpy.pi / num, 2*numpy.pi * (num-1) / num + numpy.pi / num, num)]
    else:
        return [numpy.eye(3,3) + numpy.sin(phi) * W + (1-numpy.cos(phi)) * W2 for phi in numpy.linspace(0, 2*numpy.pi * (num-1) / num, num)] 

def get_circular_offset_vectors(vec, radius, num=8, phase_shift=False):
    if radius == 0:
        return []
    perp_vec = numpy.array([-vec[1], vec[0], 0])
    circ_offset_vecs =  numpy.vstack([R.dot(normalize(perp_vec)) for R in get_rodriguies_rotations(*normalize(vec), num=num, phase_shift=phase_shift)] )
    return circ_offset_vecs * radius

def normalize(vec):
    return vec / numpy.linalg.norm(vec)

def get_square_offset_vectors(vec, width):
    vec /= numpy.linalg.norm(vec)
    assert width % 2 == 1
    
    perp_vec_1 = numpy.array([-vec[1] -vec[2], vec[0], vec[0]])
    if (numpy.linalg.norm(perp_vec_1)) < eps:
        perp_vec_1 = numpy.array([vec[2], vec[2], -vec[0] -vec[1]])
        
    perp_vec_2 = numpy.cross(vec, perp_vec_1)
    
    perp_vec_1 /= numpy.linalg.norm(perp_vec_1)
    perp_vec_2 /= numpy.linalg.norm(perp_vec_2)
    
    result = numpy.zeros((3, width, width), dtype=numpy.float32)
    for i in xrange(width):
        for j in xrange(width):
            result[:, i, j] = perp_vec_1*(i-width/2) + perp_vec_2*(j-width/2)   
    return result
    
    
    

class TrackMateReader(object):
    def __init__(self, name, filename, xyz_scale=None):
        self.name = name
        self.filename = filename
        self.data = pandas.read_csv(filename, sep="\t")
        self.xyz_scale = numpy.array([1,1,1])
        if xyz_scale is not None:
            self.xyz_scale = numpy.array(xyz_scale)
            
        
    def get_xyz_positions(self, track_id, frame=None, sortby="FRAME"):
        if frame is None:
            res =  numpy.array(self.data[(self.data["TRACK_ID"].isin(track_id))].sort(sortby)[["POSITION_X", "POSITION_Y", "POSITION_Z", "FRAME", "TRACK_ID"]]) 
        else:
            res = numpy.array(self.data[(self.data["TRACK_ID"].isin(track_id)) & (self.data["FRAME"] == frame)].sort(sortby)[["POSITION_X", "POSITION_Y", "POSITION_Z", "FRAME", "TRACK_ID"]]) 
        res[:,:3] *= self.xyz_scale
        return res
        
    def get_all_track_ids(self):
        return sorted(self.data["TRACK_ID"].unique())
    
    def get_track_ids_from_frame(self, time, sortby="TRACK_ID"):
        return numpy.array(self.data[(self.data["FRAME"] == time)].sort(sortby)["TRACK_ID"])
    
    def get_times(self):
        return sorted(self.data["FRAME"].unique())
    
    def join_on_frame(self, other):
        self_ids = self.get_all_track_ids()
        other_ids = other.get_all_track_ids()
        for own_id in self_ids:
            for own_vec in self.get_xyz_positions((own_id,)):
                frame = own_vec[3]
                for other_vec in other.get_xyz_positions(other_ids, frame):
                    other_id = int(other_vec[4])
                    yield int(frame), int(own_id), int(other_id), own_vec[:3], other_vec[:3]
    
class Kymograph3D(object):
    def __init__(self, image_filename, path_to_image, track_mate_file_origin, track_mate_file_dest, xyz_scale):
        if not os.path.exists(image_filename):
            raise IOError("File name %s does not exist" % image_filename)
        
        if not os.path.exists(track_mate_file_origin):
            raise IOError("File name %s does not exist" % track_mate_file_origin)
        
        if not os.path.exists(track_mate_file_dest):
            raise IOError("File name %s does not exist" % track_mate_file_dest)
        
        if len(xyz_scale) != 3:
            raise RuntimeError("xzy-scale has to be a list of three scalars, e.g. [1, 1, 2.35]")
        
        log.info('Initialize Kymograph3D')
        
        h = h5py.File(image_filename, "r")
        try:
            self.image_raw = h[path_to_image].value
        except:
            raise IOError("HDF5 file '%s' does not contain path to image '%s'" % (image_filename, path_to_image))
        finally:
            h.close()
            
        self.image_filename = image_filename
        self.path_to_image = path_to_image
        self.track_mate_file_origin = track_mate_file_origin
        self.track_mate_file_dest = track_mate_file_dest
        self.xyz_scale = xyz_scale
        
        log.info('Initialize TrackMate readers')
        self.track_mate_origin = TrackMateReader("Origin", track_mate_file_origin, xyz_scale)
        self.track_mate_dest = TrackMateReader("Destination", track_mate_file_dest, xyz_scale)
        
    def compute(self, radius=1, aggregation='mean', extension=(0, 0.1), on_channels=(0,1), ids=None, integration='full', 
                export_planes=False,
                plane_width=5,
                plane_pixel_width=31):
        log.info("Compute kymorgraphs for radis='%d', aggregation='%s' and extension='%r'" % (radius, aggregation, extension))
        if not (0 <= radius < 12) or not isinstance(radius, (int,)):
            raise RuntimeError("line width needs to be an integer with 0 < radius < 12")
        
        if not aggregation in ("mean", "max"):
            raise RuntimeError("line radius aggregation functions needs to be 'mean' or 'max'") 
        
        self.kymograph_vectors = defaultdict(dict)
        self.kymograph_data = defaultdict(dict)
        self.kymograph_plane_data = defaultdict(dict)
        
        for frame, k_id, p_id, k_vec, p_vec in self.track_mate_dest.join_on_frame(self.track_mate_origin):
            if (ids is not None and (p_id, k_id) in ids) or ids is None:
                log.info("\tExtracting vectors for ids: %d %d and frame %d" % (k_id, p_id, frame))
                self.kymograph_vectors[(k_id, p_id)][frame] = k_vec, p_vec
            
        for (k_id, p_id), current_kymo in self.kymograph_vectors.items():
            if extension[1] == -1:
                max_len = numpy.max(map(lambda vecs: numpy.linalg.norm(vecs[1]-vecs[0]), current_kymo.values()))
                extension[1] = max_len
            
            log.info("\tgenerating for ids: %d %d" % (k_id, p_id))
            for frame, (destination, origin) in current_kymo.items():
                self.kymograph_data[(k_id, p_id)][frame] = self._extract_line([self.image_raw[frame, c, :, :, :] for c in on_channels], 
                                                                   origin, 
                                                                   destination, 
                                                                   radius,
                                                                   aggregation,
                                                                   extension,
                                                                   integration)
                if export_planes:
                    self.kymograph_plane_data[(k_id, p_id)][frame] = self._extract_plane([self.image_raw[frame, c, :, :, :] for c in on_channels], 
                                                                           origin, 
                                                                           destination, 
                                                                           extension,
                                                                           width=plane_width,
                                                                           pixel_width=plane_pixel_width) 
                                                                         
                    
                
    def export_planes(self, channel_scaling, output_dir=".",  prefix="_planes",):
        if len(self.kymograph_plane_data) == 0:
            RuntimeError("Plane images not generated. Use the flag in compute, in order to generate them")
        log.info('Exporting plane images to folder "%s"'% os.path.abspath(output_dir))
        for k_id, p_id in self.kymograph_plane_data:
            for frame in self.kymograph_plane_data[(k_id, p_id)]:
                planes = self.kymograph_plane_data[(k_id, p_id)][frame]
                planes_c0 = planes[1]
                planes_c1 = planes[0]
                img = numpy.zeros((planes_c0.shape[0], planes_c0.shape[1], planes_c0.shape[2], 3), dtype=numpy.float32)
                for p in xrange(planes_c0.shape[2]):    
                    img[:,:,p, 0] = planes_c0[:,:,p]
                    img[:,:,p, 1] = planes_c1[:,:,p]
                for c in xrange(2):
                    img[:,:,:,c] = (img[:,:,:,c] - img[:,:,:,c].min())
                    img[:,:,:,c] *= channel_scaling[c]  
                vigra.impex.writeVolume(vigra.VigraArray(img.clip(0,255).astype(numpy.uint8), axistags=vigra.VigraArray.defaultAxistags(4)), os.path.join(output_dir, "%s_O%02d_D%05d_T%03d.tif" % (prefix, p_id, k_id, frame)), '', dtype=numpy.uint8)            
        
    def export(self, output_dir='.', prefix="kymo", channel_scaling=(1, 1)):
        log.info('Exporting kymograph images to folder "%s"'% os.path.abspath(output_dir))
        for k_id, p_id in self.kymograph_data:
            start_time = numpy.min(self.kymograph_data[(k_id, p_id)].keys())
            kymograph_img = self._create_kymogrpah_image(k_id, p_id, channel_scaling)
            vigra.impex.writeImage(kymograph_img.clip(0,255).astype(numpy.uint8), os.path.join(output_dir, "%s_O%02d_D%05d_T%03d.tif" % (prefix, p_id, k_id, start_time)), dtype=numpy.uint8)
            
    def _create_kymogrpah_image(self, k_id, p_id, channel_scaling): 
        max_len = numpy.max(map(lambda x: len(x[0]), self.kymograph_data[(k_id, p_id)].values()))
        max_time = numpy.max(self.kymograph_data[(k_id, p_id)].keys())
        kymograph_img = numpy.zeros((max_time+1, max_len, 3), dtype=numpy.float32)
        for frame in self.kymograph_data[(k_id, p_id)]:
            kymograph_img[frame, :len(self.kymograph_data[(k_id, p_id)][frame][0]), 0] = self.kymograph_data[(k_id, p_id)][frame][1]
            kymograph_img[frame, :len(self.kymograph_data[(k_id, p_id)][frame][1]), 1] = self.kymograph_data[(k_id, p_id)][frame][0]
                     
        for c in range(2):
            # some pixels get negativ
            kymograph_img[:,:,c] = (kymograph_img[:,:,c] - kymograph_img[:,:,c].min())# / (kymograph_img[:,:,0].max() - kymograph_img[:,:,0].min()) 
            kymograph_img[:,:,c] *= channel_scaling[c]   
        return kymograph_img
    
    def export_raw(self, output_dir=".", prefix="kymo_raw"):
        log.info('Exporting raw kymograph images per channel to folder "%s"'% os.path.abspath(output_dir))
        for k_id, p_id in self.kymograph_data:
            start_time = numpy.min(self.kymograph_data[(k_id, p_id)].keys())
            kymograph_img = self._create_kymogrpah_image(k_id, p_id, channel_scaling=(1,1))
            for c in range(2):
                vigra.impex.writeImage(kymograph_img.astype(numpy.float32)[:,:, c], os.path.join(output_dir, "%s_C%02d_O%02d_D%05d_T%03d.tif" % (prefix, c, p_id, k_id, start_time)), dtype=numpy.float32)
    
    def export_butterfly(self, output_dir=".", prefix="kymo_butterfly", channel_scaling=(1.2, 0.2)):
        log.info('Exporting kymograph butterfly images to folder "%s"'% os.path.abspath(output_dir))
        
        ids = set()
        for _, p_id in self.kymograph_data:
            ids.add(p_id)
        
        for i in ids:
            kymograph_img_1 = self._create_kymogrpah_image(0, i, channel_scaling)
            kymograph_img_2 = numpy.fliplr(self._create_kymogrpah_image(1, i, channel_scaling))
            butterfly_img = numpy.hstack((kymograph_img_2, kymograph_img_1))
            vigra.impex.writeImage(butterfly_img.clip(0, 255).astype(numpy.uint8), os.path.join(output_dir, "%s_O%05d.tif" % (prefix, i)), dtype=numpy.uint8)

    def _extract_plane(self, images, origin, destination, extension, width=5, pixel_width=31):
        scale_factor = float(width) / pixel_width
        origin_ext = origin + (origin - destination) * extension[0]
        if extension[1] > 1:
            num = extension[1] 
            dest_ext = origin + normalize(destination - origin) * num 
            assert (num - numpy.linalg.norm(origin-dest_ext)) < 10e-10
        else:
            dest_ext = destination + (destination - origin) * extension[1]
            num = numpy.linalg.norm(dest_ext-origin_ext)
        num = numpy.linalg.norm(dest_ext-origin_ext)
    
        x =  numpy.linspace(origin_ext[0], dest_ext[0], num)
        y =  numpy.linspace(origin_ext[1], dest_ext[1], num)
        z =  numpy.linspace(origin_ext[2], dest_ext[2], num)
        
        vec = destination-origin
        vec /= numpy.linalg.norm(vec)
        plane_cords = get_square_offset_vectors(vec, pixel_width)          
        plane_cords.shape += (1,) 
        plane_cords = numpy.repeat(plane_cords, len(x), 3)
       
        plane_cords*= scale_factor      
        plane_cords[2, :, :, :] += x
        plane_cords[1, :, :, :] += y
        plane_cords[0, :, :, :] += z
        
        planes = [ndimage.map_coordinates(img, plane_cords, prefilter=False,  mode='nearest', cval=0) for img in images]
        return planes
        

    def _extract_line(self, images, origin, destination, radius, aggregation, extension, integration='full'):
        #origin_ext = destination - 5*(origin - destination)/(origin - destination)
        origin_ext = origin + (origin - destination) * extension[0]
        if extension[1] > 1:
            num = extension[1] 
            dest_ext = origin + normalize(destination - origin) * num 
            assert (num - numpy.linalg.norm(origin-dest_ext)) < 10e-10
        else:
            dest_ext = destination + (destination - origin) * extension[1]
            num = numpy.linalg.norm(dest_ext-origin_ext)
        num = numpy.linalg.norm(dest_ext-origin_ext)
    
        x =  numpy.linspace(origin_ext[0], dest_ext[0], num)
        y =  numpy.linspace(origin_ext[1], dest_ext[1], num)
        z =  numpy.linspace(origin_ext[2], dest_ext[2], num)
        
        if integration=="full":
            
            all_circ_offsets = []
            for r in range(1, radius+1):
                perp_offset_vecs = get_circular_offset_vectors(destination-origin, r, num=r*8, phase_shift=not bool(r % 2))
                all_circ_offsets.append(perp_offset_vecs)
            if len(all_circ_offsets) > 0:
                perp_offset_vecs = numpy.vstack(all_circ_offsets) 
            else:
                perp_offset_vecs = []   
            coords = numpy.zeros((3, len(x), len(all_circ_offsets)+1))
            coords[2, :, -1] = x
            coords[1, :, -1] = y
            coords[0, :, -1] = z
        elif integration=='rim':
            if radius == 0:
                RuntimeError("For integration = 'rim', radius has to be > 0")
            perp_offset_vecs = get_circular_offset_vectors(destination-origin, radius, phase_shift=False)
            coords = numpy.zeros((3, len(x), len(perp_offset_vecs)))
        else:
            RuntimeError("Integration %s not understood. not in ('full', 'rim')" % integration)
            
        
        
        for j, (x_,y_,z_) in enumerate(perp_offset_vecs):
            coords[2, :, j] = x_ + x
            coords[1, :, j] = y_ + y
            coords[0, :, j] = z_ + z
        
        if aggregation == 'max':
            profiles = [ndimage.map_coordinates(img, coords, prefilter=False,  mode='nearest', cval=0).max(1) for img in images]
        elif aggregation == 'mean':
            profiles = [ndimage.map_coordinates(img, coords, prefilter=False, mode='nearest', cval=0).mean(1) for img in images]
        else:
            RuntimeError("width_aggregation not understood: '%s'" % aggregation)
        return profiles
    
class Kymograph3DTestBase(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

class Kymograph3DTestBasic(Kymograph3DTestBase):  
    pass

def test_rodriguez_rot(length=20, radius=3, phase_shift=False):
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        
        vec = numpy.array([4,8,6])
        
        x, y, z = numpy.linspace(0, vec[0], length), numpy.linspace(0, vec[1], length), numpy.linspace(0, vec[2], length)
        for r in range(1, radius+1):
            if phase_shift:
                ov = get_circular_offset_vectors(vec, r, not bool(r % 2))
            else:
                ov = get_circular_offset_vectors(vec, r, False)
            for k, (xi, yi, zi) in enumerate(zip(x, y, z)):
                ax.plot([xi], [yi], [zi], 'ro')
                for j, a in enumerate(ov):
                    ax.plot([a[0] + xi], [a[1]+ yi], [a[2]+zi], 'o', color=(0,0.2,k/float(length)))
        ax.axis("equal")
        plt.show()
        
def test_square_sampling(width=5, length=40):
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        
        vec = numpy.array([1.0, 10.0, 40.0])
        
        x, y, z = numpy.linspace(0, vec[0], length), numpy.linspace(0, vec[1], length), numpy.linspace(0, vec[2], length)

        ov = get_square_offset_vectors(vec, width)
        for k, (xi, yi, zi) in enumerate(zip(x, y, z)):
            ax.plot([xi], [yi], [zi], 'rd')
            for x_ in range(ov.shape[1]):
                for y_ in range(ov.shape[2]):
                    ax.plot([ov[0, x_, y_]+xi], [ov[1, x_, y_]+ yi], [ov[2, x_, y_]+zi], 'bo',)
        #ax.axis("equal")
        plt.show()
    
def convert_and_resample_from_tif(file_name, output_file='image_cropped_ana.h5', z_factor=2.35, path_to_image='data'):
    import javabridge
    import bioformats
    
    javabridge.start_vm(class_path=bioformats.JARS)
    r = bioformats.ImageReader(file_name)
    
    shape = (r.rdr.getSizeT(), r.rdr.getSizeC(), r.rdr.getSizeZ(), r.rdr.getSizeY(), r.rdr.getSizeX())
    shape_r = (r.rdr.getSizeT(), r.rdr.getSizeC(), int(z_factor * r.rdr.getSizeZ()), r.rdr.getSizeY(), r.rdr.getSizeX())
    
    img = numpy.zeros(shape, dtype=numpy.float32)
    img_r = numpy.zeros(shape_r, dtype=numpy.float32)
    img_r_prefilter = numpy.zeros(shape_r, dtype=numpy.float32)
    for t in range(shape[0]):
        print "T:", t,
        for c in range(shape[1]):
            for z in range(shape[2]):
                img[t, c, z, :, :,] = r.read(c=c, t=t, z=z)
            print ".",
            img_r[t,c,:,:,:] = vigra.sampling.resizeVolumeSplineInterpolation(img[t,c,:,:,:], shape_r[2:])
            img_r_prefilter[t, c, :, :, :] = ndimage.spline_filter(img_r[t,c,:,:,:])
        
    f = h5py.File(output_file, 'w')
    f["/"].create_dataset(path_to_image, data=img)
    f["/"].create_dataset(path_to_image + "_resampled", data=img_r)
    f["/"].create_dataset(path_to_image + "_resampled_prefiltered", data=img_r_prefilter)
    f.close()
    javabridge.kill_vm()  
    
if __name__ == "__main__":
    if False:
        convert_and_resample_from_tif('M:/experiments/Experiments_002500/002513/Analysis/2014_03_27(microtubules)/Process_02_3D_tracking_CENPA/cell1_12_crop_3D_Gaussian_ROI_t11_150.tif',
                                      'cell1_12_crop.h5',
                                      z_factor=2.35)
        
    #test_rodriguez_rot(length=100,radius=1)
         
    if True:
        kymo = Kymograph3D("cell1_12_crop.h5",
#                             "data_resampled_prefiltered",
                            "data_resampled",
                            "tracks_kinetochore.txt",
                            "tracks_pole.txt",
                            [1, 1, 2.35],)
#         kymo.compute(radius=0, aggregation='mean', extension=[0,-1], ids=((0,24), (1,24)))
        kymo.compute(radius=0, aggregation='mean', extension=[0,-1])
        kymo.export_butterfly(channel_scaling=(2, 0.25))
        kymo.export(channel_scaling=(2, 0.25))

        
                   
        
import sys
import numpy as np
from sklearn.cluster import DBSCAN
import h5py

class Domain:
    """
    TODO:
        pre_filtering:
            has problem, do not use, need check
    """
    def __init__(self, flnm=None, data=None, data_shape=None):
        if flnm is not None:
            self.flnm = flnm
        elif data is not None:
            self.data = data
            self.data_shape = data_shape
            self.index = np.arange(np.prod(data_shape))
        else:
            sys.exit("flnm and data can not be None at the same time")
        self.contour = None
        self.regions = []
        self.outputed_region = 0
    @property
    def num_points(self):
        return len(self.index)
    @property
    def num_regions(self):
        return len(self.regions)
    @property
    def num_bounded_regions(self):
        num=0
        for region in self.regions:
            if region.bounded:
                num+=1
        return num
    @property
    def exist_contour(self):
        return bool(self.contour)
    def set_sound_speed(self, snd):
        self.sound_speed = snd

    def reset_Domain(self):
        self.contour = None
        self.regions = []

    def _subset(self, sub):
        self.contour = Contour(sub)
        return self.contour
    def thresholding(self, threshold, field='grav_pot'):
        data = self.data[field][self.index]
        sub = np.argwhere(data>threshold).ravel()
        if sub.sum() == 0:
            print("Domain: No point satisfy the condition, skipping "\
                    "[thresholding]")
            self.reset_Domain()
            return False
        self._subset(sub)
        return True 
    def dbscan(self, **kwds):
        if not self.exist_contour:
            print("Domain: Domain have no countor, skippting [dbcan]")
            return False
        self.regions = self.contour._dbscan(self.data,self.data_shape, **kwds)
        return True
    def check_boundness(self, **kwds):
        if self.num_regions == 0:
            print("Domain: Domain have no regions, skippting [check_boundness]")
            return False
        if 'sound_speed' not in kwds.keys():
            if hasattr(self,'sound_speed'):
                kwds['sound_speed'] = self.sound_speed
        for region in self.regions:
            region._check_boundness(self.data, **kwds)
        return True
    def _open_h5(self, initialize = False):
        dset_name = 'bounded_region'
        if initialize:
            hf = h5py.File('result.h5', 'w')
            empty_data = np.zeros(self.data_shape)
            hf.create_dataset(dset_name, data=empty_data)
        else:
            hf = h5py.File('result.h5', 'a')
        return hf
    def output_bounded_region(self):
        if self.num_bounded_regions == 0:
            print("Domain: Domain have no bounded regions, skippting "\
                    "[output_bounded_region]")
            return False
        if self.outputed_region == 0:
            f = self._open_h5(initialize=True)
        else:
            f = self._open_h5()
        dset = f['bounded_region']
        for region in self.regions:
            if not region.bounded:
                continue
            self.outputed_region += 1
            region_index = self.outputed_region
            positions = np.asarray([self.data['pos_%s' % idir][region.index]
                for idir in 'ijk']).T
            for pos in positions:
                dset[tuple(pos)] = region_index
        f.close()
        return True
    def subtract_bounded_region(self):
        if not self.regions:
            print("Domain: Domain have no regions, skippting "\
                    "[subtract_bounded_region]")
            return False
        bounded_region = []
        for region in self.regions:
            if region.bounded:
                bounded_region+=list(region.index)
        self.index = np.asarray(list(set(self.index)-set(bounded_region)))
        return True
    def print_summary(self):
        print("""
        Summary:
          # of points = {},
          # of regions = {},
          # of bounded regions = {}
        """.format(self.num_points, self.num_regions, self.num_bounded_regions))

    def _sequence(self, threshold, dbscan_args=dict(),
            boundness_args=dict()):
        self.thresholding(threshold)
        self.dbscan(**dbscan_args)
        self.check_boundness(**boundness_args)
        self.output_bounded_region()
        self.subtract_bounded_region()
        self.print_summary()

    def operate(self,threshold, **kwargs):
        if isinstance(threshold, (list, np.ndarray)):
            threshold = np.asarray(threshold)
            threshold.sort()
            for th in threshold:
                self._sequence(th, **kwargs)
        else:
            self._sequence(th, **kwargs)

    def load_zeus(self):
        data = {}
        keys = ['gas_density', 'grav_pot',  #'gas_energy',
                'i_mag_field', 'j_mag_field', 'k_mag_field', 
                'i_velocity',  'j_velocity',  'k_velocity'
                ]
        with h5py.File(self.flnm,"r") as f:
            # for key in list(f.keys()):
            for key in keys:
                data[key] = f[key][()].ravel()
            data_shape = f[key].shape
        data['pos_i'],data['pos_j'],data['pos_k'] = np.meshgrid(*[np.arange(i)
            for i in data_shape],indexing = 'ij')
        for idir in "ijk":
            data['pos_%s' % idir] = data['pos_%s' % idir].ravel()
        self.data = data
        self.data_shape = data_shape
        self.index = np.arange(np.prod(data_shape))
        
    def pre_filter(self):
        pass

    def dump_data(self):
        import pickle
        parameters = ['data_shape','index', 'sound_speed']
        for para in parameters:
            if hasattr(self,para):
                self.data[para] = getattr(self, para)
        a_file = open("preloaded_data.pkl", "wb")
        pickle.dump(self.data, a_file)
        a_file.close()
    def load_data(self):
        import pickle
        parameters = ['data_shape', 'index', 'sound_speed']
        a_file = open('preloaded_data.pkl', 'rb')
        self.data = pickle.load(a_file)
        for para in parameters:
            setattr(self, para, self.data[para])
            self.data.pop(para)
        a_file.close()
        self.index = np.arange(len(self.data['grav_pot']))


class Region:
    """
    TODO:
        Now, assuming uniform grid, and assuming dv = 1 for all grid.
        Therefore, density = mass. Need more work for AMR data.
    """
    def __init__(self, index):
        self.index = index
    @property
    def num_points(self):
        return len(self.index)

    def _kinetic_energy(self, data):
        """
        calculate kinetic energy for given data within the region.
        """
        dset_name = 'gas_density'
        if dset_name in data.keys():
            density = data[dset_name][self.index]
        else:
            density = 1.
        v2 = 0
        for idir in 'ijk':
            dset_name = "%s_velocity" % idir
            if dset_name in data.keys():
                velocity = data[dset_name][self.index]
                velocity -= velocity.mean()
                v2 += velocity**2
        kinetic_ene = (0.5*density*v2).sum()
        return kinetic_ene
    def _magnetic_energy(self, data):
        b2 = 0
        for idir in 'ijk':
            dset_name = "%s_mag_field" % idir
            if dset_name in data.keys():
                b2 +=  data[dset_name][self.index]**2
        magnetic_ene = (0.5*b2).sum()
        return magnetic_ene
    def _thermal_energy(self, data, sound_speed):
        dset_name = 'gas_density'
        density = data[dset_name][self.index]
        thermal_ene = (density*sound_speed**2).sum()
        return thermal_ene

    def _potential_energy(self, data):
        dset_name = 'gas_density'
        density = data[dset_name][self.index]
        dset_name = 'grav_pot'
        grav_pot = data[dset_name][self.index]
        # Note the sign of input potential has been inverted
        pot_ene = -(density*grav_pot).sum()
        return pot_ene

    def _check_boundness(self, data, fields = 
            ['kinetic_energy','magnetic_enenrgy','thermal_energy'],
            sound_speed = None):
        total_ene = 0
        # count kinetic energy density
        if 'kinetic_energy' in fields:
            total_ene += self._kinetic_energy(data)
        # count magnetic energy density
        if 'magnetic_energy' in fields:
            total_ene += self._magnetic_energy(data)
        # count thermal energy density
        if ('thermal_energy' in fields and 
            sound_speed is not None):
            total_ene += self._thermal_energy(data, sound_speed)
        elif 'thermal_energy' in fields:
            print ("Please define sound speed")
            sys.exit()
        # count potential enenrgy density, 
        total_ene += self._potential_energy(data)

        # if total enerngy is negative, the region is gravitationally bounded
        if total_ene <= 0:
            self.bounded = True
        else:
            self.bounded = False

class Contour(Region):
    def _subset(self, sub):
        return Region(sub)

    def _split(self, sub_sets):
        regions = []
        for sub in sub_sets:
            regions.append(self._subset(sub))
        return regions

    def _dbscan(self,data,data_shape,eps=1.8,min_samp=5,wrapped=True, n_jobs = 2):
        position = np.asarray([data['pos_%s' % p][self.index] for p in 'ijk']).T
        # define wrapped euclidiean metric
        def wrapped_euclidean_points(p, q):
            diff = np.abs(p - q)
            return np.linalg.norm(np.minimum(diff, data_shape - diff))
        # apply DBSCAN to data
        if wrapped == True:
            db = DBSCAN(eps=eps,min_samples=min_samp,
                metric=wrapped_euclidean_points,
                n_jobs=n_jobs).fit(position)
        else:
            db = DBSCAN(eps=eps, min_samples=min_samp, n_jobs=n_jobs).fit(position)
        # distinguish samples and noise, get label of samples
        core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
        core_samples_mask[db.core_sample_indices_] = True
        labels = db.labels_
        # Number of clusters in labels, ignoring noise if present.
        self.n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        self.n_noise = list(labels).count(-1)
        # put regions to blankets
        unique_labels = set(labels) - set([-1])
        sub_sets = []
        for k in unique_labels:
            class_member_mask = (labels == k)
            sub_sets.append(self.index[class_member_mask
                & core_samples_mask])
        sub_regions = self._split(sub_sets)
        return sub_regions

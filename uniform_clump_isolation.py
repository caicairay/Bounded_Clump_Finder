import sys
import numpy as np
import h5py
import cc3d

class Domain:
    """
    TODO:
        pre_filtering:
            may have problem, need check
    """
    def __init__(self, flnm=None, data=None, data_shape = None):
        if flnm is not None:
            self.flnm = flnm
            self.output_flnm = flnm+"_result.h5"
        elif data is not None:
            self.data = data
            self.data_shape = data_shape
            self.valid_domain = np.ones(data_shape)
            self.output_flnm = "result.h5"
        else:
            sys.exit("flnm and data can not be None at the same time")
        self.outputed_region = 0

    def set_sound_speed(self, snd):
        self.sound_speed = snd

    def thresholding(self, threshold, field='grav_pot', layers=1):
        data_main = self.data[field]
        labels_in_main = np.logical_and(data_main > threshold, self.valid_domain)
        labels_out_main = cc3d.connected_components(labels_in_main) # 26-connected
        N = np.max(labels_out_main)
        # Dealing with boundary condition
        indices = np.arange(-layers,layers)
        labels_in = [np.take(labels_in_main,indices,axis=idir) for idir in range(3)]
        for idir in range(3):
            labels_out = cc3d.connected_components(labels_in[idir]) # 26-connected
            # if no labels at boundary, skip
            if labels_out.sum() == 0: continue
            # if there are labels at boundary
            labels = np.unique(labels_out)
            for label in labels:
                ends = labels_out == label
                end1 = np.take(ends, range(0,layers), axis = idir)
                end2 = np.take(ends, range(-layers,0), axis = idir)
                domain_end1 = np.take(labels_out_main,range(-layers,0),axis=idir) 
                domain_end2 = np.take(labels_out_main,range(0,layers), axis=idir) 
                domain_label1 = np.unique(domain_end1[end1])
                domain_label2 = np.unique(domain_end2[end2])
                labels_out_main[np.isin(labels_out_main, domain_label1)] = N+label
                labels_out_main[np.isin(labels_out_main, domain_label2)] = N+label
            N = np.max(labels_out_main)
        self.labels_out = labels_out_main
        self.labels = np.unique(labels_out_main)

    def _kinetic_energy(self, region):
        """
        calculate kinetic energy for given data within the region.
        """
        dset_name = 'gas_density'
        density = self.data[dset_name][region]
        v2 = 0
        for idir in 'ijk':
            dset_name = "%s_velocity" % idir
            if dset_name in self.data.keys():
                velocity = self.data[dset_name][region]
                velocity -= velocity.mean()
                v2 += velocity**2
        kinetic_ene = (0.5*density*v2).sum()
        return kinetic_ene
    def _magnetic_energy(self, region):
        b2 = 0
        for idir in 'ijk':
            dset_name = "%s_mag_field" % idir
            if dset_name in self.data.keys():
                b2 +=  self.data[dset_name][region]**2
        magnetic_ene = (0.5*b2).sum()
        return magnetic_ene
    def _thermal_energy(self, region):
        dset_name = 'gas_density'
        density = self.data[dset_name][region]
        thermal_ene = (density*self.sound_speed**2).sum()
        return thermal_ene
    def _potential_energy(self, region):
        dset_name = 'gas_density'
        density = self.data[dset_name][region]
        dset_name = 'grav_pot'
        grav_pot = self.data[dset_name][region]
        # Note the sign of input potential has been inverted
        pot_ene = -(density*grav_pot).sum()
        return pot_ene
    def check_boundness(self):
        self.bounded_labels = []
        for label in self.labels:
            total_ene = 0
            region = self.labels_out == label
            total_ene += self._kinetic_energy(region)
            total_ene += self._magnetic_energy(region)
            total_ene += self._thermal_energy(region)
            total_ene += self._potential_energy(region)
            if total_ene < 0:
                self.bounded_labels.append(label)
    def _open_h5(self, initialize = False):
        dset_name = 'bounded_region'
        if initialize:
            hf = h5py.File(self.output_flnm, 'w')
            empty_data = np.zeros(self.data_shape)
            hf.create_dataset(dset_name, data=empty_data)
        else:
            hf = h5py.File(self.output_flnm, 'a')
        return hf
    def output_bounded_region(self):
        if self.outputed_region == 0:
            f = self._open_h5(initialize=True)
        else:
            f = self._open_h5()
        dset = f['bounded_region']
        for label in self.bounded_labels:
            self.outputed_region += 1
            region = self.labels_out == label
            region_index = self.outputed_region
            dset[region] = region_index
        f.close()
    def subtract_bounded_region(self):
        for label in self.bounded_labels:
            region = self.labels_out == label
            self.valid_domain[region] = False

    def load_zeus(self):
        keys = ['gas_density', 'grav_pot',  #'gas_energy',
                'i_mag_field', 'j_mag_field', 'k_mag_field', 
                'i_velocity',  'j_velocity',  'k_velocity'
                ]
        with h5py.File(self.flnm,"r") as f:
            data = {key: f[key][()] for key in keys}
        self.data = data
        self.data_shape = data['grav_pot'].shape
        self.valid_domain = np.ones(self.data_shape, dtype = bool)

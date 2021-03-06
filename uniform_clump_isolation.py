import numpy as np
import h5py
import cc3d

def CCL(labels_in_main, layers=1):
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
        labels = np.unique(labels_out)[1:]
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
    return labels_out_main

class Domain:
    """
    TODO:
        None
    """
    def __init__(self, flnm=None, data=None, data_shape = None):
        if flnm is not None:
            self.flnm = flnm
            self.output_flnm = flnm+"_result.h5"
        elif data is not None:
            self.data = data
            self.data_shape = data_shape
            self.valid_domain = np.ones(data_shape, dtype=np.int)
            self.output_flnm = "result.h5"
        else:
            sys.exit("flnm and data can not be None at the same time")
        self.outputed_region = 0

    def set_sound_speed(self, snd):
        self.sound_speed = snd

    def thresholding(self, threshold, field='grav_pot'):
        data_main = self.data[field]
        labels_in_main = data_main > threshold
        labels_out_main = CCL(labels_in_main)
        self.labels_out = labels_out_main
        self.labels = np.unique(labels_out_main)[1:]

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
        pot_ene = (density*grav_pot).sum()
        return pot_ene
    def check_boundness(self):
        self.bounded_labels = []
        self.ready_to_output_labels = []
        for label in self.labels:
            total_ene = 0
            region = self.labels_out == label
            # if the region contains outputed structures, skip
            if 0 in self.valid_domain[region]:
                continue
            # the region has no outputed structures, check boundness
            total_ene += self._kinetic_energy(region)
            total_ene += self._magnetic_energy(region)
            total_ene += self._thermal_energy(region)
            total_ene -= self._potential_energy(region)
            # if the region is bounded, mark it
            if total_ene <= 0.:
                self.bounded_labels.append(label)
            # if the region is not bounded, check if the region
            # contains bounded structure.
            else:
                if 2 in self.valid_domain[region]:
                    self.ready_to_output_labels.append(label)
                else:
                    pass
    def _open_h5(self, dset_name, output_flnm, initialize = False):
        if initialize:
            hf = h5py.File(output_flnm, 'w')
            empty_data = np.zeros(self.data_shape)
            hf.create_dataset(dset_name, data=empty_data)
        else:
            hf = h5py.File(output_flnm, 'a')
        return hf
    def output_bounded_region(self):
        dset_name = 'bounded_region'
        if self.outputed_region == 0:
            f = self._open_h5(dset_name, self.output_flnm, initialize=True)
        else:
            f = self._open_h5(dset_name, self.output_flnm)
        dset = f[dset_name]
        for label in self.ready_to_output_labels:
            self.outputed_region += 1
            region = self.labels_out == label
            region_index = self.outputed_region
            dset[region] = region_index
        f.close()
    def update_region_status(self):
        for label in self.ready_to_output_labels:
            region = self.labels_out == label
            self.valid_domain[region] = 0
        for label in self.bounded_labels:
            region = self.labels_out == label
            self.valid_domain[region] = 2

    def load_zeus(self):
        keys = ['gas_density', 'grav_pot',  #'gas_energy',
                'i_mag_field', 'j_mag_field', 'k_mag_field', 
                'i_velocity',  'j_velocity',  'k_velocity'
                ]
        with h5py.File(self.flnm,"r") as f:
            data = {key: f[key][()] for key in keys}
        self.data = data
        self.data_shape = data['grav_pot'].shape
        self.valid_domain = np.ones(self.data_shape, dtype = np.int)

    def _cut_edges(self, threshold, label):
        labels_in_main = np.logical_and(self.data['grav_pot'] >= threshold, 
                                   self.result == label)
        labels_out_main = CCL(labels_in_main)
        record = {}
        for i in np.unique(labels_out_main)[1:]:
            record[i] = (labels_out_main == i).sum()
        saved_label = max(record, key=record.get)
        self.cutted_region = labels_out_main == saved_label
           
    def _calculate_ratio(self, region):
        total_ene = 0
        total_ene += self._kinetic_energy(region)
        total_ene += self._magnetic_energy(region)
        total_ene += self._thermal_energy(region)
        return total_ene/self._potential_energy(region)
    
    def _search_potential(self, low, high, label, tolerance, max_steps):
        i = 0
        while (i <= max_steps):
            mid = (low+high)/2.
            self._cut_edges(mid, label)
            ratio = self._calculate_ratio(self.cutted_region)
            print('ratio={}, label={},mid={}'.format(ratio, label, mid))
            if ratio >= 1.-tolerance and ratio < 1.: 
                return mid
            elif ratio >= 1.:
                low = mid
            elif ratio < 1.-tolerance:
                high = mid
            i += 1
        return mid

    def retrive_potential(self, seq, tolerance = 0.05, max_steps = 50):
        """
        FOR label IN labels:
            select region
            determine potential range
            search potential
            update result
        """
        with h5py.File(self.output_flnm, "r") as f:
            self.result = f["bounded_region"][()]
        labels = np.arange(1, self.outputed_region + 1)
        for label in labels:
            region = self.result == label
            pot_min = self.data['grav_pot'][region].min()
            pot_max = seq[seq>pot_min].min()
            pot_retrived = self._search_potential(pot_min, pot_max, label,
                    tolerance, max_steps)
            selection = self.cutted_region
            # retrive region
            region[selection] = False
            self.result[region] = 0
        retrived_flnm = self.flnm+"_retrived_result.h5"
        dset_name = 'retrived_region'
        f = self._open_h5(dset_name, retrived_flnm, initialize = True)
        dset = f[dset_name]
        dset[()] = self.result
        f.close()

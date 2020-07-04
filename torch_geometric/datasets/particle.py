import os.path as osp
import glob

import torch
import pandas
import numpy as np
from torch_geometric.data import Data, Dataset


class TrackMLParticleTrackingDataset(Dataset):
    r"""The `TrackML Particle Tracking Challenge
    <https://www.kaggle.com/c/trackml-particle-identification>`_ dataset to
    reconstruct particle tracks from 3D points left in the silicon detectors.

    Args:
        root (string): Root directory where the dataset should be saved.

        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)

        volume_layer_ids (List): List of the volume and layer ids to be included
            in the graph. Layers get indexed by increasing volume and layer id.

        layer_pairs (List): List of which pairs of layers can have edges between them

        pt_min (float32): A truth cut applied to reduce the number of nodes in the graph.
            Only nodes associated with particles above this momentum are included.

        eta_range ([min, max]): A cut applied to nodes to select a specific eta

        phi_slope_max (float32): A cut applied to edges to limit the change in phi between
            the two nodes.

        z0_max (float32): A cut applied to edges that limits how far from the center of
            the detector the particle edge can originate from.

        n_phi_sections (int): Break the graph into multiple segments in the phi direction.

        n_eta_sections (int): Break the graph into multiple segments in the eta direction.

    """

    url = 'https://www.kaggle.com/c/trackml-particle-identification'

    def __init__(self, root, transform=None,
                 volume_layer_ids=[[8, 2], [8, 4], [8, 6], [8, 8]], #Layers Selected
                 layer_pairs=[[0, 1], [1, 2], [2, 3]],              #Connected Layers
                 pt_min=2.0, eta_range=[-5, 5],                     #Node Cuts
                 phi_slope_max=0.0006, z0_max=150,                  #Edge Cuts
                 n_phi_sections=1, n_eta_sections=1                 #N Sections
                 ):
        events = glob.glob(osp.join(osp.join(root, 'raw'), 'event*-hits.csv'))
        events = [e.split(osp.sep)[-1].split('-')[0][5:] for e in events]
        self.events = sorted(events)

        self.volume_layer_ids = torch.tensor(volume_layer_ids)
        self.layer_pairs      = torch.tensor(layer_pairs)
        self.pt_min           = pt_min
        self.eta_range        = eta_range
        self.phi_slope_max    = phi_slope_max
        self.z0_max           = z0_max
        self.n_phi_sections   = n_phi_sections
        self.n_eta_sections   = n_eta_sections

        super(TrackMLParticleTrackingDataset, self).__init__(root, transform)


    @property
    def raw_file_names(self):
        if not hasattr(self,'input_files'):
            self.input_files = sorted(glob.glob(self.raw_dir+'/*.csv'))
        return [f.split('/')[-1] for f in self.input_files]


    @property
    def processed_file_names(self):
        if not hasattr(self,'processed_files'):
            proc_names = ['data_{}.pt'.format(idx) for idx in self.events]
            self.processed_files = [osp.join(self.processed_dir,name) for name in proc_names]
        return self.processed_files


    def download(self):
        raise RuntimeError(
            'Dataset not found. Please download it from {} and move all '
            '*.csv files to {}'.format(self.url, self.raw_dir))


    def len(self):
        return len(glob.glob(osp.join(self.raw_dir, 'event*-hits.csv')))


    def __len__(self):
        return len(glob.glob(osp.join(self.raw_dir, 'event*-hits.csv')))


    def read_hits(self, idx):
        hits_filename = osp.join(self.raw_dir, f'event{idx}-hits.csv')
        hits = pandas.read_csv(
            hits_filename, usecols=['hit_id', 'x', 'y', 'z', 'volume_id', 'layer_id', 'module_id'],
            dtype={
                'hit_id': np.int64,
                'x': np.float32,
                'y': np.float32,
                'z': np.float32,
                'volume_id': np.int64,
                'layer_id': np.int64,
                'module_id': np.int64
            })
        return hits


    def read_cells(self, idx):
        cells_filename = osp.join(self.raw_dir, f'event{idx}-cells.csv')
        cells = pandas.read_csv(
            cells_filename, usecols=['hit_id', 'ch0', 'ch1', 'value'],
            dtype={
                'hit_id': np.int64,
                'ch0': np.int64,
                'ch1': np.int64,
                'value': np.float32
            })
        return cells


    def read_particles(self, idx):
        particles_filename = osp.join(self.raw_dir, f'event{idx}-particles.csv')
        particles = pandas.read_csv(
            particles_filename, usecols=['particle_id', 'vx', 'vy', 'vz', 'px', 'py', 'pz', 'q', 'nhits'],
            dtype={
                'particle_id': np.int64,
                'vx': np.float32,
                'vy': np.float32,
                'vz': np.float32,
                'px': np.float32,
                'py': np.float32,
                'pz': np.float32,
                'q': np.int64,
                'nhits': np.int64
            })
        return particles


    def read_truth(self, idx):
        truth_filename = osp.join(self.raw_dir, f'event{idx}-truth.csv')
        truth = pandas.read_csv(
            truth_filename, usecols=['hit_id', 'particle_id', 'tx', 'ty', 'tz', 'tpx', 'tpy', 'tpz', 'weight'],
            dtype={
                'hit_id': np.int64,
                'particle_id': np.int64,
                'tx': np.float32,
                'ty': np.float32,
                'tz': np.float32,
                'tpx': np.float32,
                'tpy': np.float32,
                'tpz': np.float32,
                'weight': np.float32
            })
        return truth


    def select_hits(self, hits, particles, truth):
        # print('Selecting Hits')
        valid_layer = 20 * self.volume_layer_ids[:,0] + self.volume_layer_ids[:,1]
        hits = (hits[['hit_id', 'x', 'y', 'z', 'volume_id', 'layer_id']]
                .merge(truth[['hit_id', 'particle_id']], on='hit_id'))
        hits = (hits[['hit_id', 'x', 'y', 'z', 'volume_id', 'layer_id', 'particle_id']]
                .merge(particles[['particle_id', 'px', 'py', 'pz']], on='particle_id'))

        layer = torch.from_numpy(20 * hits['volume_id'].values + hits['layer_id'].values)
        r = torch.from_numpy(np.sqrt(hits['x'].values**2 + hits['y'].values**2))
        phi = torch.from_numpy(np.arctan2(hits['y'].values, hits['x'].values))
        z = torch.from_numpy(hits['z'].values)
        theta = torch.atan2(r,z)
        eta = -1*torch.log(torch.tan(theta/2))
        pt = torch.from_numpy(np.sqrt(hits['px'].values**2 + hits['py'].values**2))
        particle = torch.from_numpy(hits['particle_id'].values)

        layer_mask = torch.from_numpy(np.isin(layer, valid_layer))
        eta_mask1 = eta > self.eta_range[0]
        eta_mask2 = eta < self.eta_range[1]
        pt_mask = pt > self.pt_min
        mask = layer_mask & eta_mask1 & eta_mask2 & pt_mask

        r = r[mask]
        phi = phi[mask]
        z = z[mask]
        pos = torch.stack([r, phi, z], 1)
        layer = layer[mask].unique(return_inverse=True)[1]
        particle = particle[mask]

        layer, indices = torch.sort(layer)
        pos = pos[indices]
        particle = particle[indices]

        return pos, layer, particle


    def compute_edge_index(self, pos, layer):
        # print("Constructing Edge Index")
        for (layer1, layer2) in self.layer_pairs:
            mask1 = layer == layer1
            mask2 = layer == layer2
            nnz1 = mask1.nonzero().flatten()
            nnz2 = mask2.nonzero().flatten()

            dr   = pos[:, 0][mask2].view(1, -1) - pos[:, 0][mask1].view(-1, 1)
            dphi = pos[:, 1][mask2].view(1, -1) - pos[:, 1][mask1].view(-1, 1)
            dz   = pos[:, 2][mask2].view(1, -1) - pos[:, 2][mask1].view(-1, 1)
            dphi[dphi > np.pi] -= 2 * np.pi
            dphi[dphi < -np.pi] += 2 * np.pi

            phi_slope = dphi / dr
            z0 = pos[:, 2][mask1].view(-1, 1) - pos[:, 0][mask1].view(-1, 1) * dz / dr

            adj = (phi_slope.abs() < self.phi_slope_max) & (z0.abs() < self.z0_max)

            row, col = adj.nonzero().t()
            row = nnz1[row]
            col = nnz2[col]
            edge_index = torch.stack([row, col], dim=0)

            if (layer1 == self.layer_pairs[0,0]):
                edge_indices = edge_index
            else:
                edge_indices = torch.cat((edge_indices, edge_index), 1)

        return edge_indices


    def compute_y_index(self, edge_indices, particle):
        # print("Constructing y Index")
        pid1 = [ particle[i].item() for i in edge_indices[0] ]
        pid2 = [ particle[i].item() for i in edge_indices[1] ]
        y = np.zeros(edge_indices.shape[1], dtype=np.int64)
        for i in range(edge_indices.shape[1]):
            if pid1[i] == pid2[i]:
                y[i] = 1

        return torch.from_numpy(y)


    def read_event(self, idx):
        hits      = self.read_hits(idx)
        # cells     = self.read_cells(idx)
        particles = self.read_particles(idx)
        truth     = self.read_truth(idx)

        return hits, particles, truth


    def process(self):
        for idx in self.events:
            hits, particles, truth = self.read_event(idx)
            pos, layer, particle = self.select_hits(hits, particles, truth)
            edge_index = self.compute_edge_index(pos, layer)
            y = self.compute_y_index(edge_index, particle)

            # if self.pre_filter is not None and not self.pre_filter(data):
            #     continue
            #
            # if self.pre_transform is not None:
            #     data = self.pre_transform(data)

            data = Data(x=pos, edge_index=edge_index, y=y)
            torch.save(data, osp.join(self.processed_dir, 'data_{}.pt'.format(idx)))


    def get(self, idx):
        data = torch.load(self.processed_files[idx])
        return data

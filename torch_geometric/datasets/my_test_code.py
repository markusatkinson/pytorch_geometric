from particle import TrackMLParticleTrackingDataset
from torch_geometric.data import Data, Dataset, InMemoryDataset
import argparse
import yaml


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser('my_test_code.py')
    add_arg = parser.add_argument
    add_arg('config', nargs='?', default='configs/prepare_trackml.yaml')
    return parser.parse_args()


def main():
    """Main function"""
    # Parse the command line
    args = parse_args()

    with open(args.config) as f:
        config = yaml.load(f)
        selection = config['selection']

    trackml_data = TrackMLParticleTrackingDataset('/data/gnn_code/trackml-particle-identification/',
                                                  volume_layer_ids=selection['volume_layer_ids'],
                                                  layer_pairs=selection['layer_pairs'],
                                                  pt_min=selection['pt_min'],
                                                  eta_range=selection['eta_range'],
                                                  phi_slope_max=selection['phi_slope_max'],
                                                  z0_max=selection['z0_max'],
                                                  augments=selection['construct_augmented_graphs'],
                                                  intersect=selection['remove_intersecting_edges'],
                                                  )
    # print(trackml_data.raw_file_names)
    # print(trackml_data.processed_file_names)
    # print(trackml_data.input_files)
    # print(trackml_data.volume_layer_ids)
    # print(trackml_data.layer_pairs)
    # print(trackml_data.pt_min)
    # print(trackml_data.eta_range)
    # print(trackml_data.phi_slope_max)
    # print(trackml_data.z0_max)

    print(trackml_data.augments)
    print(trackml_data.intersect)
    trackml_data.process()
    # print(trackml_data.processed_file_names)

    print(trackml_data[50])
    print(trackml_data[51])
    print(trackml_data[150])
    print(trackml_data[151])
    # print(trackml_data.num_features)
    # print(trackml_data.processed_file_names)

if __name__ == '__main__':
    main()

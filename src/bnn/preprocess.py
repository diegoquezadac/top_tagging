import h5py
import numpy as np

def compute_stats(file_path, max_constits=80):
    """Compute dataset-level sums for fjet_clus_pt and fjet_clus_E, ignoring padded constituents."""
    sum_pt = 0.0
    sum_energy = 0.0
    batch_size = 100_000
    data_vector_names = ["fjet_clus_pt", "fjet_clus_E"]
    
    with h5py.File(file_path, 'r', swmr=True) as h5file:
        num_samples = len(h5file['labels'])
        
        for start in range(0, num_samples, batch_size):
            end = min(start + batch_size, num_samples)
            data_dict = {key: h5file[key][start:end] for key in data_vector_names}
            
            # Extract raw data, truncated to max_constits
            pt = data_dict['fjet_clus_pt'][:, :max_constits]
            energy = data_dict['fjet_clus_E'][:, :max_constits]
            
            # Mask for padded constituents (pt == 0)
            mask = (pt == 0)
            
            # Sum only real (non-padded) constituents
            sum_pt += np.sum(pt[~mask])
            sum_energy += np.sum(energy[~mask])
    
    return sum_pt, sum_energy

def constituent(data_dict, max_constits=80, sum_pt_global=None, sum_energy_global=None):
    """Constituent function that normalizes pt and energy by global sums if provided, else per-jet sums, keeping 7 features."""
    # Pull data from data dict
    pt = data_dict['fjet_clus_pt'][:, :max_constits]
    eta = data_dict['fjet_clus_eta'][:, :max_constits]
    phi = data_dict['fjet_clus_phi'][:, :max_constits]
    energy = data_dict['fjet_clus_E'][:, :max_constits]

    # Find location of zero pt entries
    mask = np.asarray(pt == 0).nonzero()

    # Angular preprocessing
    eta_shift = eta[:, 0]
    phi_shift = phi[:, 0]
    eta -= eta_shift[:, np.newaxis]
    phi -= phi_shift[:, np.newaxis]
    phi = np.where(phi > np.pi, phi - 2 * np.pi, phi)
    phi = np.where(phi < -np.pi, phi + 2 * np.pi, phi)
    second_eta = eta[:, 1]
    second_phi = phi[:, 1]
    alpha = np.arctan2(second_phi, second_eta) + np.pi / 2
    eta_rot = (eta * np.cos(alpha[:, np.newaxis]) + phi * np.sin(alpha[:, np.newaxis]))
    phi_rot = (-eta * np.sin(alpha[:, np.newaxis]) + phi * np.cos(alpha[:, np.newaxis]))
    eta = eta_rot
    phi = phi_rot
    third_eta = eta[:, 2]
    parity = np.where(third_eta < 0, -1, 1)
    eta = (eta * parity[:, np.newaxis]).astype(np.float32)
    radius = np.sqrt(eta ** 2 + phi ** 2)

    # pT and energy features
    log_pt = np.log(pt)
    log_energy = np.log(energy)
    
    # Normalize by global sums if provided, else per-jet sums
    if sum_pt_global is not None and sum_energy_global is not None:
        lognorm_pt = np.log(pt / sum_pt_global)
        lognorm_energy = np.log(energy / sum_energy_global)
    else:
        sum_pt = np.sum(pt, axis=1)
        sum_energy = np.sum(energy, axis=1)
        lognorm_pt = np.log(pt / sum_pt[:, np.newaxis])
        lognorm_energy = np.log(energy / sum_energy[:, np.newaxis])

    # Reset padded entries to zero
    eta[mask] = 0
    phi[mask] = 0
    log_pt[mask] = 0
    log_energy[mask] = 0
    lognorm_pt[mask] = 0
    lognorm_energy[mask] = 0
    radius[mask] = 0

    # Stack along last axis (7 features)
    features = [eta, phi, log_pt, log_energy, lognorm_pt, lognorm_energy, radius]
    stacked_data = np.stack(features, axis=-1)

    return stacked_data

def preprocess(file_path, output_path, max_constits=80, batch_size=500000, use_train_weights=False):
    """Create a preprocessed HDF5 file with globally normalized pt and energy features."""
    data_vector_names = ['fjet_clus_pt', 'fjet_clus_eta', 'fjet_clus_phi', 'fjet_clus_E']
    
    # First pass: Compute dataset-level sums
    sum_pt_global, sum_energy_global = compute_stats(file_path, max_constits)
    
    # Second pass: Preprocess and write in batches
    with h5py.File(file_path, 'r', swmr=True) as input_file:
        num_samples = len(input_file['labels'])
        
        with h5py.File(output_path, 'w') as output_file:

            features_ds = output_file.create_dataset('features', shape=(num_samples, max_constits, 7), dtype=np.float32)
            labels_ds = output_file.create_dataset('labels', shape=(num_samples,), dtype=input_file['labels'].dtype)
            if use_train_weights:
                weights_ds = output_file.create_dataset('weights', shape=(num_samples,), dtype=input_file['weights'].dtype)
            
            for start in range(0, num_samples, batch_size):
                end = min(start + batch_size, num_samples)
                data_dict = {key: input_file[key][start:end] for key in data_vector_names}

                
                # Apply preprocessing with global normalization
                processed_data = constituent(data_dict, max_constits, 
                                          sum_pt_global=sum_pt_global, 
                                          sum_energy_global=sum_energy_global)
                
                features_ds[start:end] = processed_data
                labels_ds[start:end] = input_file['labels'][start:end]
                if use_train_weights:
                    weights_ds[start:end] = input_file['weights'][start:end]
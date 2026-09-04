import os
import sys
import time
import logging
import builtins
import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data, Dataset
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
from torch.optim.swa_utils import AveragedModel, SWALR
from ase import Atoms
from ase.neighborlist import neighbor_list
from torch_scatter import scatter
from collections import defaultdict


os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
sys.path.insert(0, current_dir)

_original_torch_load = torch.load
def _hack_torch_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)
torch.load = _hack_torch_load

try:
    torch.serialization.add_safe_globals([builtins.slice])
except: pass

def setup_logger(name, log_file=None, level=logging.INFO):
    formatter = logging.Formatter(fmt='%(asctime)s %(levelname)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if not logger.handlers:
        logger.addHandler(handler)
    if log_file:
        file_handler = logging.FileHandler(log_file, mode='w')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger

logger = setup_logger("MACE_Train", "output_final.log")

try:
    from module.get_dataset import TrajectoryDataset
    from model_delta import MaceModel
except ImportError as e:
    logger.error(f"Error importing modules: {e}")
    sys.exit(1)


EXPERIMENT_NAME = "water_delta"
BEST_MODEL_PATH = f'best_model_{EXPERIMENT_NAME}.pth'
BEST_EMA_MODEL_PATH = f'best_model_{EXPERIMENT_NAME}_ema.pth'
SEED = 98

LEARNING_RATE = 0.001
WEIGHT_DECAY = 5e-7
AMSGRAD = True
BATCH_SIZE = 16
GRADIENT_ACCUM_STEPS = 1
NUM_EPOCHS = 150
FORCE_WEIGHT = 1000.0
NUM_WORKERS = 16

EMA_DECAY = 0.99
USE_EMA = True

LR_SCHEDULER_PATIENCE = 5
LR_SCHEDULER_FACTOR = 0.5
MIN_LR_FOR_EARLY_STOPPING = 1e-6

NUM_ATOM_TYPES = 3
FEATURE_IRREPS_HIDDEN = "128x0e + 128x1o + 128x2e"
CORRELATION = 3
NUM_INTERACTIONS = 2
N_SCALAR = 128
RADIAL_DIM = 8
RADIAL_WIDTH = 64
EDGE_ATTR_LMAX = 2
LMAX_CENTER = 2
LMAX_ENV = 2
CUTOFF_RADIUS = 5.0


DATA_NPZ_PATH = os.path.join(parent_dir, 'dataset', 'V_3.npz')
DATASET_ROOT = os.path.join(current_dir, 'processed_v3')
PBC_SETTING = [True, True, False]

VAL_RATIO = 0.1
TEST_RATIO = 0.1


VALID_INDICES_PATH = "valid_indices_water.txt"
TEST_INDICES_PATH = "test_indices_water.txt"


def set_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class InMemoryDataset(Dataset):
    def __init__(self, data_list):
        super().__init__()
        self.data_list = data_list
    def len(self):
        return len(self.data_list)
    def get(self, idx):
        return self.data_list[idx]

def compute_avg_neighbors(dataset, indices, cutoff):
    logger.info("Computing average number of neighbors...")
    if len(indices) == 0: return 34.0
    num_samples = min(len(indices), 50)
    sample_indices = np.random.choice(indices, num_samples, replace=False)
    total_neighbors = 0
    total_atoms = 0
    for i in tqdm(sample_indices, desc="Computing Neighbors", leave=False):
        data = dataset[i]
        total_neighbors += data.edge_index.shape[1]
        total_atoms += data.num_nodes
    return total_neighbors / total_atoms if total_atoms > 0 else 0

def compute_forces_rms(dataset, indices):
    logger.info("Computing Forces RMS...")
    if len(indices) == 0: return 1.0
    squared_forces = []
    sample_indices = np.random.choice(indices, min(len(indices), 100), replace=False)
    for i in tqdm(sample_indices, desc="Computing RMS", leave=False):
        data = dataset[i]
        f = data.force_target
        is_center = data.is_center
        f_real = f[is_center] if is_center.any() else f
        squared_forces.append(f_real.pow(2).mean().item())
    return np.sqrt(np.mean(squared_forces))


def build_graph_correct_pbc(data, cutoff=5.0):
    pos = data.pos.clone() if isinstance(data.pos, torch.Tensor) else torch.from_numpy(data.pos).float()
    atomic_numbers = data.x.numpy() if hasattr(data.x, 'numpy') else data.x
    cell = data.cell.numpy() if hasattr(data.cell, 'numpy') else data.cell

    if cell.shape == (3,): cell = np.diag(cell)
    elif cell.shape != (3, 3): cell = cell.reshape(3, 3)

    num_atoms = len(atomic_numbers)
    atoms = Atoms(numbers=atomic_numbers, positions=pos.numpy(), cell=cell, pbc=PBC_SETTING)
    i_idx, j_idx, S = neighbor_list('ijS', atoms, cutoff)

    mask = i_idx != j_idx
    i_idx = i_idx[mask]
    j_idx = j_idx[mask]
    S = S[mask]

    edge_index = torch.stack([torch.from_numpy(i_idx), torch.from_numpy(j_idx)], dim=0).long()
    S_tensor = torch.from_numpy(S).float()
    cell_tensor = torch.from_numpy(cell).float()

    if S_tensor.dim() == 1:
        if S_tensor.numel() == 0: edge_shift = torch.zeros((0, 3))
        else: S_tensor = S_tensor.view(-1, 3); edge_shift = S_tensor @ cell_tensor
    else: edge_shift = S_tensor @ cell_tensor
    if edge_shift.dim() == 1: edge_shift = edge_shift.view(-1, 3)

    z_map = {1: 0, 3: 1, 8: 2}
    mapped_x = data.x.clone()
    for z_real, z_mapped in z_map.items():
        mapped_x[data.x == z_real] = z_mapped
    mapped_x = mapped_x.long()

    is_center = torch.zeros(num_atoms, dtype=torch.bool)
    num_electrode = 36
    if num_atoms > 2 * num_electrode:
        is_center[num_electrode:-num_electrode] = True
    else:
        is_center[:] = True

    raw_vbias = data.vbias
    v_val = raw_vbias.item() if hasattr(raw_vbias, 'item') else float(raw_vbias)
    voltage_scalar = torch.tensor([[v_val]], dtype=torch.float)

    return Data(
        x=mapped_x,
        atom_type=mapped_x,
        voltage=voltage_scalar,
        pos=pos,
        edge_index=edge_index,
        edge_shift=edge_shift,
        is_center=is_center,
        cell=data.cell,
        energy_target=data.energy_target,
        force_target=data.force_target,
        node_type=is_center.long()
    )


def loss_fn(pred_energy, true_energy, pred_forces, true_forces, is_center, force_weight, forces_rms):
    energy_loss = torch.nn.functional.mse_loss(pred_energy, true_energy.squeeze())
    central_pred = pred_forces[is_center]
    central_true = true_forces[is_center]
    if central_pred.numel() > 0:
        force_loss = torch.nn.functional.mse_loss(central_pred, central_true) / (forces_rms ** 2)
    else:
        force_loss = torch.tensor(0.0, device=pred_energy.device)
    return energy_loss + force_weight * force_loss

def evaluate(model, loader, device, forces_rms):
    model.eval()
    total_loss, total_energy_ae, total_energy_se, total_energy_se_per_atom = 0.0, 0.0, 0.0, 0.0
    total_force_ae, total_force_se = 0.0, 0.0
    num_atoms_total, num_force_samples, num_graphs = 0, 0, 0

    with torch.enable_grad():
        for i, batch in enumerate(loader):
            batch = batch.to(device)
            num_graphs += batch.num_graphs
            num_atoms_total += batch.num_nodes

            with torch.amp.autocast('cuda'):
                batch.pos.requires_grad_(True)
                pred_energy, pred_forces = model(batch)

            pred_energy = pred_energy.detach().float()
            pred_forces = pred_forces.detach().float()
            true_energy = batch.energy_target
            true_forces = batch.force_target

            loss = loss_fn(pred_energy, true_energy, pred_forces, true_forces, batch.is_center, FORCE_WEIGHT, forces_rms)
            total_loss += loss.item()

            energy_diff = pred_energy.view(-1) - true_energy.view(-1)
            total_energy_ae += torch.sum(torch.abs(energy_diff)).item()
            total_energy_se += torch.sum(energy_diff ** 2).item()

            n_atoms_per_graph = scatter(torch.ones_like(batch.batch), batch.batch, reduce='sum')
            total_energy_se_per_atom += torch.sum((energy_diff / n_atoms_per_graph) ** 2).item()

            is_center = batch.is_center
            if is_center.any():
                f_pred = pred_forces[is_center]
                f_true = true_forces[is_center]
                f_diff = f_pred - f_true
                total_force_ae += torch.sum(torch.abs(f_diff)).item()
                total_force_se += torch.sum(f_diff ** 2).item()
                num_force_samples += f_pred.numel()

    metrics = {
        'loss': total_loss / len(loader),
        'mae_e': (total_energy_ae / num_graphs) * 1000,
        'rmse_e': (torch.sqrt(torch.tensor(total_energy_se / num_graphs)) * 1000).item(),
        'mae_e_per_atom': (total_energy_ae / num_atoms_total) * 1000,
        'rmse_e_per_atom': (torch.sqrt(torch.tensor(total_energy_se_per_atom / num_graphs)) * 1000).item(),
        'mae_f': (total_force_ae / num_force_samples) * 1000 if num_force_samples else 0.0,
        'rmse_f': (torch.sqrt(torch.tensor(total_force_se / num_force_samples)) * 1000).item() if num_force_samples else 0.0,
    }
    return metrics


def main():
    set_seed(SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    logger.info("===========VERIFYING SETTINGS===========")
    logger.info(f"Experiment: {EXPERIMENT_NAME}")

    logger.info("===========LOADING INPUT DATA===========")
    graph_builder = lambda data: build_graph_correct_pbc(data, cutoff=CUTOFF_RADIUS)

    logger.info(f"Loading dataset from {DATA_NPZ_PATH}...")
    disk_dataset = TrajectoryDataset(root=DATASET_ROOT, file_path=DATA_NPZ_PATH, transform=graph_builder)

    logger.info("Pre-processing and caching dataset in RAM...")
    caching_batch_size = 64
    loader_for_caching = DataLoader(disk_dataset, batch_size=caching_batch_size, shuffle=False, num_workers=NUM_WORKERS, pin_memory=False)
    cached_data_list = []
    for batch in tqdm(loader_for_caching, desc="Caching Graphs", file=sys.stderr):
        cached_data_list.extend(batch.to_data_list())

    dataset = InMemoryDataset(cached_data_list)
    all_indices = np.arange(len(dataset))

    logger.info("===========SPLITTING DATASET===========")
    indices_exist = os.path.exists(VALID_INDICES_PATH) and os.path.exists(TEST_INDICES_PATH)

    if indices_exist:
        logger.info(f"Loading fixed indices...")
        val_indices = np.loadtxt(VALID_INDICES_PATH, dtype=int)
        test_indices = np.loadtxt(TEST_INDICES_PATH, dtype=int)
        exclude_indices = np.concatenate([val_indices, test_indices])
        train_indices = np.setdiff1d(all_indices, exclude_indices)
    else:
        logger.info("Extracting signatures for double-stratification...")
        signatures = [(tuple(torch.bincount(data.x, minlength=3).tolist()[:3]), round(data.voltage.item(), 2)) for data in cached_data_list]
        grouped_indices = defaultdict(list)
        for idx, sig in enumerate(signatures):
            grouped_indices[sig].append(idx)

        train_indices_list, val_indices_list, test_indices_list = [], [], []
        for sig, idxs in grouped_indices.items():
            idxs = np.array(idxs)
            np.random.shuffle(idxs)
            n_total = len(idxs)
            n_val = max(1, int(np.floor(VAL_RATIO * n_total))) if n_total >= 3 else 0
            n_test = max(1, int(np.floor(TEST_RATIO * n_total))) if n_total >= 3 else 0
            val_indices_list.extend(idxs[:n_val])
            test_indices_list.extend(idxs[n_val : n_val + n_test])
            train_indices_list.extend(idxs[n_val + n_test:])

        train_indices, val_indices, test_indices = np.array(train_indices_list, dtype=int), np.array(val_indices_list, dtype=int), np.array(test_indices_list, dtype=int)
        np.random.shuffle(train_indices)
        np.random.shuffle(val_indices)
        np.random.shuffle(test_indices)
        np.savetxt(VALID_INDICES_PATH, val_indices, fmt='%d')
        np.savetxt(TEST_INDICES_PATH, test_indices, fmt='%d')

    logger.info(f"Split counts: Train={len(train_indices)}, Valid={len(val_indices)}, Test={len(test_indices)}")
    avg_neighbors = compute_avg_neighbors(dataset, train_indices, CUTOFF_RADIUS)
    forces_rms = compute_forces_rms(dataset, train_indices)

    train_loader = DataLoader(torch.utils.data.Subset(dataset, train_indices), batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(torch.utils.data.Subset(dataset, val_indices), batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    test_loader = DataLoader(torch.utils.data.Subset(dataset, test_indices), batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    logger.info("===========MODEL DETAILS===========")
    model = MaceModel(
        n_scalar=N_SCALAR,
        num_atom_types=NUM_ATOM_TYPES,
        num_interactions=NUM_INTERACTIONS,
        correlation=CORRELATION,
        feature_irreps_hidden=FEATURE_IRREPS_HIDDEN,
        radial_dim=RADIAL_DIM,
        radial_width=RADIAL_WIDTH,
        edge_attr_lmax=EDGE_ATTR_LMAX,
        avg_num_neighbors=avg_neighbors,
        rbf_cutoff=CUTOFF_RADIUS,
        lmax_center=LMAX_CENTER,
        lmax_env=LMAX_ENV,
        num_electrode=36,
        electrolyte_types=()
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total number of parameters: {total_params}")

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY, amsgrad=AMSGRAD)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=LR_SCHEDULER_FACTOR, patience=LR_SCHEDULER_PATIENCE)
    scaler = GradScaler()
    ema_model = AveragedModel(model, multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(EMA_DECAY)) if USE_EMA else None

    logger.info("===========TRAINING===========")
    best_val_loss = float('inf')

    for epoch in range(NUM_EPOCHS):
        model.train()
        train_loss_epoch = 0.0
        optimizer.zero_grad()
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}", leave=False, file=sys.stderr)

        for step, batch in enumerate(progress_bar):
            batch = batch.to(device)
            with torch.amp.autocast('cuda'):
                batch.pos.requires_grad_(True)
                pred_energy, pred_forces = model(batch)
                loss = loss_fn(pred_energy.squeeze(), batch.energy_target, pred_forces, batch.force_target, batch.is_center, FORCE_WEIGHT, forces_rms)
                loss = loss / GRADIENT_ACCUM_STEPS

            scaler.scale(loss).backward()

            if (step + 1) % GRADIENT_ACCUM_STEPS == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                if USE_EMA: ema_model.update_parameters(model)

            train_loss_epoch += loss.item() * GRADIENT_ACCUM_STEPS
            progress_bar.set_postfix({'loss': f"{loss.item() * GRADIENT_ACCUM_STEPS:.4f}"})

        eval_model = ema_model if USE_EMA else model
        val_metrics = evaluate(eval_model, val_loader, device, forces_rms)

        logger.info(f"Epoch {epoch}: loss={val_metrics['loss']:.4f}, RMSE_E={val_metrics['rmse_e']:.2f} meV, MAE_E={val_metrics['mae_e']:.2f} meV, RMSE_F={val_metrics['rmse_f']:.2f} meV/A, MAE_F={val_metrics['mae_f']:.2f} meV/A")

        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            torch.save(ema_model.module.state_dict() if USE_EMA else model.state_dict(), BEST_EMA_MODEL_PATH if USE_EMA else BEST_MODEL_PATH)

        scheduler.step(val_metrics['loss'])
        if optimizer.param_groups[0]['lr'] < MIN_LR_FOR_EARLY_STOPPING:
            logger.info("Learning rate below minimum threshold. Stopping early.")
            break

    logger.info("===========FINAL TESTING===========")
    target_model_path = BEST_EMA_MODEL_PATH if USE_EMA else BEST_MODEL_PATH
    state_dict = torch.load(target_model_path, map_location=device)
    model.load_state_dict(state_dict)

    test_metrics = evaluate(model, test_loader, device, forces_rms)
    logger.info(f"Test Loss: {test_metrics['loss']:.4f} | Test RMSE_E: {test_metrics['rmse_e']:.4f} meV | Test RMSE_F: {test_metrics['rmse_f']:.4f} meV/A")

if __name__ == '__main__':
    main()

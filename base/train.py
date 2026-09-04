import os
import sys
import time
import logging
import builtins
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data, Dataset
from tqdm import tqdm
from torch.nn.functional import cosine_similarity
from torch.cuda.amp import autocast, GradScaler
from torch.optim.swa_utils import AveragedModel, SWALR
from ase import Atoms
from ase.neighborlist import neighbor_list
_TRAIN_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_TRAIN_DIR)
sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, _TRAIN_DIR)


os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


def setup_logger(name, log_file=None, level=logging.INFO):
    formatter = logging.Formatter(
        fmt='%(asctime)s %(levelname)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
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

logger = setup_logger("MACE_Train", "output_standard_training.log")


try:
    from module.get_dataset import TrajectoryDataset
    from model import MaceModel
except ImportError as e:
    logger.error(f"Error importing modules: {e}")
    sys.exit(1)


_original_torch_load = torch.load
def _hack_torch_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)
torch.load = _hack_torch_load
try:
    torch.serialization.add_safe_globals([builtins.slice])
except: pass


EXPERIMENT_NAME = "data3"
BEST_MODEL_PATH = f'best_model_{EXPERIMENT_NAME}.pth'
BEST_EMA_MODEL_PATH = f'best_model_{EXPERIMENT_NAME}_ema.pth'
SEED = 1234


LEARNING_RATE = 0.005
WEIGHT_DECAY = 5e-7
AMSGRAD = True
BATCH_SIZE = 8
GRADIENT_ACCUM_STEPS = 1
NUM_EPOCHS = 300
FORCE_WEIGHT = 1000.0
NUM_WORKERS = 0


EMA_DECAY = 0.99
USE_EMA = True


LR_SCHEDULER_PATIENCE = 10
LR_SCHEDULER_FACTOR = 0.5
MIN_LR_FOR_EARLY_STOPPING = 1e-6


NUM_ATOM_TYPES = 3
FEATURE_IRREPS_HIDDEN = "128x0e + 128x1o + 128x2e"
CORRELATION = 3
NUM_INTERACTIONS = 3
N_SCALAR = 128
RADIAL_DIM = 12
RADIAL_WIDTH = 64
EDGE_ATTR_LMAX = 2
LMAX_CENTER = 2
LMAX_ENV = 2


CUTOFF_RADIUS = 5.0
DATA_NPZ_PATH = os.path.join(_REPO_ROOT, 'dataset', '0V_3.npz')
DATASET_ROOT = os.path.join(_TRAIN_DIR, 'processed_0V')
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1
TRAIN_INDICES_PATH = "train_indices_811.txt"
VAL_INDICES_PATH = "val_indices_811.txt"
TEST_INDICES_PATH = "test_indices_811.txt"
PBC_SETTING = [True, True, False]


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

def compute_average_e0(dataset, indices, num_atom_types):
    logger.info("Computing average Atomic Energies...")
    if not os.path.exists(DATA_NPZ_PATH):
        return torch.zeros(num_atom_types)
    with np.load(DATA_NPZ_PATH, allow_pickle=True) as data:
        all_atoms = data['atoms']
        all_energies = data['energy']
    z_map = {1: 0, 3: 1, 8: 2}
    len_train = len(indices)
    A = np.zeros((len_train, num_atom_types))
    b = np.zeros(len_train)
    for idx_i, i in enumerate(tqdm(indices, desc="Building Linear System", leave=False)):
        z_raw = all_atoms[i]
        z_mapped = np.zeros_like(z_raw)
        for z_real, z_idx in z_map.items():
            z_mapped[z_raw == z_real] = z_idx
        counts = np.bincount(z_mapped, minlength=num_atom_types)
        A[idx_i] = counts
        b[idx_i] = all_energies[i]
    solution, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
    return torch.from_numpy(solution).float()

def compute_avg_neighbors(dataset, indices, cutoff):
    logger.info(f"Computing average number of neighbors...")
    num_samples = min(len(indices), 100)
    sample_indices = np.random.choice(indices, num_samples, replace=False)
    total_neighbors = 0
    total_atoms = 0
    for i in sample_indices:
        data = dataset[i]
        total_neighbors += data.edge_index.shape[1]
        total_atoms += data.num_nodes
    return total_neighbors / total_atoms

def compute_forces_rms(dataset, indices):
    logger.info("Computing Forces RMS...")
    squared_forces = []
    sample_indices = np.random.choice(indices, min(len(indices), 100), replace=False)
    for i in sample_indices:
        data = dataset[i]
        f = data.force_target
        squared_forces.append(f.pow(2).mean().item())
    return np.sqrt(np.mean(squared_forces))


def build_graph_correct_pbc(data, cutoff=5.0):
    pos = data.pos
    z_raw = data.x.numpy()
    cell = data.cell.numpy()

    if cell.shape == (3,): cell = np.diag(cell)
    elif cell.shape != (3, 3): cell = cell.reshape(3, 3)

    num_atoms = len(z_raw)


    atoms = Atoms(numbers=z_raw, positions=pos.numpy(), cell=cell, pbc=PBC_SETTING)
    i_idx, j_idx, S = neighbor_list('ijS', atoms, cutoff)
    mask = i_idx != j_idx
    i_idx, j_idx, S = i_idx[mask], j_idx[mask], S[mask]

    edge_index = torch.stack([torch.from_numpy(i_idx), torch.from_numpy(j_idx)], dim=0).long()
    edge_shift = torch.from_numpy(S).float() @ torch.from_numpy(cell).float()

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

    vbias = data.vbias
    if isinstance(vbias, torch.Tensor):
        vbias = vbias.view(-1)
        target_dim = 3
        current_dim = vbias.shape[0]
        if current_dim < target_dim:
            padding = torch.zeros(target_dim - current_dim, device=vbias.device, dtype=vbias.dtype)
            vbias = torch.cat([vbias, padding], dim=0)
        elif current_dim > target_dim:
            vbias = vbias[:target_dim]
        vbias = vbias.view(1, 3)
        vbias_node = vbias.repeat(num_atoms, 1)
    else:
        vbias_node = torch.zeros((num_atoms, 3), device=data.pos.device)

    return Data(
        x=mapped_x, atom_type=mapped_x, vbias=vbias_node, pos=data.pos,
        edge_index=edge_index, edge_shift=edge_shift, is_center=is_center,
        cell=data.cell, energy_target=data.energy_target,
        force_target=data.force_target, node_type=is_center.long()
    )


def loss_fn(pred_energy, true_energy, pred_forces, true_forces, batch, force_weight, forces_rms):

    energy_loss = F.mse_loss(pred_energy.squeeze(), true_energy.squeeze(), reduction='mean')


    is_center = batch.is_center
    central_pred, central_true = pred_forces[is_center], true_forces[is_center]

    if central_pred.numel() > 0:
        force_loss = F.mse_loss(central_pred, central_true, reduction='mean')
    else:
        force_loss = torch.tensor(0.0, device=pred_energy.device)

    return energy_loss + force_weight * force_loss


def evaluate(model, loader, device, forces_rms):
    model.eval()
    total_loss, total_energy_ae, total_energy_se = 0.0, 0.0, 0.0
    total_force_ae, total_force_se = 0.0, 0.0
    num_atoms_total, num_force_samples, num_graphs = 0, 0, 0

    with torch.enable_grad():
        for batch in loader:
            batch = batch.to(device)
            num_graphs += batch.num_graphs
            num_atoms_total += batch.num_nodes

            batch.pos.requires_grad_(True)

            with torch.amp.autocast('cuda'):
                pred_energy, pred_forces = model(batch)

            pred_energy = pred_energy.detach().float().squeeze()
            pred_forces = pred_forces.detach().float()

            loss_val = loss_fn(pred_energy, batch.energy_target, pred_forces, batch.force_target,
                              batch, FORCE_WEIGHT, forces_rms)
            total_loss += loss_val.item()

            total_energy_ae += torch.abs(pred_energy - batch.energy_target).sum().item()
            total_energy_se += ((pred_energy - batch.energy_target) ** 2).sum().item()

            is_center = batch.is_center
            if is_center.any():
                f_pred, f_true = pred_forces[is_center], batch.force_target[is_center]

                total_force_ae += torch.abs(f_pred - f_true).sum().item()
                total_force_se += ((f_pred - f_true) ** 2).sum().item()
                num_force_samples += f_pred.numel()

    metrics = {
        'loss': total_loss / len(loader),
        'mae_e': (total_energy_ae / num_graphs) * 1000,
        'rmse_e': (np.sqrt(total_energy_se / num_graphs) * 1000) if num_graphs else 0.0,
        'mae_f': (total_force_ae / num_force_samples) * 1000 if num_force_samples else 0.0,
        'rmse_f': (np.sqrt(total_force_se / num_force_samples) * 1000) if num_force_samples else 0.0,
    }
    return metrics


def main():
    set_seed(SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    logger.info("=========== SETTINGS SUMMARY ===========")
    logger.info(f"Experiment: {EXPERIMENT_NAME}")
    logger.info(f"Loss Function: MSE Loss")
    logger.info(f"Data Split: Train={TRAIN_RATIO}, Val={VAL_RATIO}, Test={TEST_RATIO}")


    logger.info("=========== LOADING DATA ===========")
    graph_builder = lambda data: build_graph_correct_pbc(data, cutoff=CUTOFF_RADIUS)
    disk_dataset = TrajectoryDataset(root=DATASET_ROOT, file_path=DATA_NPZ_PATH, transform=graph_builder)

    logger.info(f"Pre-processing and caching dataset...")
    cached_data_list = [data for data in tqdm(disk_dataset, desc="Caching Graphs")]
    dataset = InMemoryDataset(cached_data_list)


    all_indices = np.arange(len(dataset))
    logger.info(f"Dataset: Total={len(dataset)} frames")


    if (os.path.exists(TRAIN_INDICES_PATH) and
        os.path.exists(VAL_INDICES_PATH) and
        os.path.exists(TEST_INDICES_PATH)):
        logger.info(f"Loading fixed indices from existing files...")
        train_indices = np.loadtxt(TRAIN_INDICES_PATH, dtype=int)
        val_indices = np.loadtxt(VAL_INDICES_PATH, dtype=int)
        test_indices = np.loadtxt(TEST_INDICES_PATH, dtype=int)
    else:
        logger.info(f"Creating new train/val/test split...")
        np.random.shuffle(all_indices)
        n_train = int(len(all_indices) * TRAIN_RATIO)
        n_val = int(len(all_indices) * VAL_RATIO)

        train_indices = all_indices[:n_train]
        val_indices = all_indices[n_train:n_train + n_val]
        test_indices = all_indices[n_train + n_val:]


        np.savetxt(TRAIN_INDICES_PATH, train_indices, fmt='%d')
        np.savetxt(VAL_INDICES_PATH, val_indices, fmt='%d')
        np.savetxt(TEST_INDICES_PATH, test_indices, fmt='%d')
        logger.info(f"Saved indices to {TRAIN_INDICES_PATH}, {VAL_INDICES_PATH}, {TEST_INDICES_PATH}")

    logger.info(f"Split: Train={len(train_indices)}, Val={len(val_indices)}, Test={len(test_indices)}")


    atomic_energies = compute_average_e0(dataset, train_indices, NUM_ATOM_TYPES)
    avg_neighbors = compute_avg_neighbors(dataset, train_indices, CUTOFF_RADIUS)
    forces_rms = compute_forces_rms(dataset, train_indices)


    train_subset = torch.utils.data.Subset(dataset, train_indices)
    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(torch.utils.data.Subset(dataset, val_indices), batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(torch.utils.data.Subset(dataset, test_indices), batch_size=BATCH_SIZE, shuffle=False)


    logger.info("Building MACE model...")
    model = MaceModel(
        n_scalar=N_SCALAR, num_atom_types=NUM_ATOM_TYPES, atomic_energies_mean=atomic_energies,
        num_interactions=NUM_INTERACTIONS, correlation=CORRELATION, feature_irreps_hidden=FEATURE_IRREPS_HIDDEN,
        radial_dim=RADIAL_DIM, radial_width=RADIAL_WIDTH, edge_attr_lmax=EDGE_ATTR_LMAX,
        avg_num_neighbors=avg_neighbors, rbf_cutoff=CUTOFF_RADIUS, lmax_center=LMAX_CENTER, lmax_env=LMAX_ENV
    ).to(device)


    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY, amsgrad=AMSGRAD)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=LR_SCHEDULER_FACTOR, patience=LR_SCHEDULER_PATIENCE)
    scaler = GradScaler()
    ema_model = AveragedModel(model, multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(EMA_DECAY)) if USE_EMA else None


    best_val_loss = float('inf')

    for epoch in range(NUM_EPOCHS):
        model.train()
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}", leave=False, file=sys.stderr)

        for step, batch in enumerate(progress_bar):
            batch = batch.to(device)
            with torch.amp.autocast('cuda'):
                batch.pos.requires_grad_(True)
                pred_energy, pred_forces = model(batch)
                loss = loss_fn(pred_energy, batch.energy_target, pred_forces, batch.force_target,
                               batch, force_weight=FORCE_WEIGHT, forces_rms=forces_rms)
                loss = loss / GRADIENT_ACCUM_STEPS

            scaler.scale(loss).backward()

            if (step + 1) % GRADIENT_ACCUM_STEPS == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                if USE_EMA: ema_model.update_parameters(model)

        eval_model = ema_model if USE_EMA and ema_model.n_averaged > 0 else model
        val_metrics = evaluate(eval_model, val_loader, device, forces_rms)

        log_msg = (
            f"Epoch {epoch}: Loss={val_metrics['loss']:.4f}, "
            f"MAE_E={val_metrics['mae_e']:.2f} meV, RMSE_E={val_metrics['rmse_e']:.2f} meV, "
            f"MAE_F={val_metrics['mae_f']:.2f} meV/Å, RMSE_F={val_metrics['rmse_f']:.2f} meV/Å"
        )
        logger.info(log_msg)

        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            save_path = BEST_EMA_MODEL_PATH if USE_EMA else BEST_MODEL_PATH
            torch.save((ema_model.module if USE_EMA else model).state_dict(), save_path)

        scheduler.step(val_metrics['loss'])
        if optimizer.param_groups[0]['lr'] < MIN_LR_FOR_EARLY_STOPPING:
            logger.info("Early stopping due to learning rate reduction.")
            break


    logger.info("=========== EVALUATING ON TEST SET ===========")
    best_model_path = BEST_EMA_MODEL_PATH if USE_EMA else BEST_MODEL_PATH
    if os.path.exists(best_model_path):
        logger.info(f"Loading best model from {best_model_path}")
        test_model = MaceModel(
            n_scalar=N_SCALAR, num_atom_types=NUM_ATOM_TYPES, atomic_energies_mean=atomic_energies,
            num_interactions=NUM_INTERACTIONS, correlation=CORRELATION, feature_irreps_hidden=FEATURE_IRREPS_HIDDEN,
            radial_dim=RADIAL_DIM, radial_width=RADIAL_WIDTH, edge_attr_lmax=EDGE_ATTR_LMAX,
            avg_num_neighbors=avg_neighbors, rbf_cutoff=CUTOFF_RADIUS, lmax_center=LMAX_CENTER, lmax_env=LMAX_ENV
        ).to(device)
        test_model.load_state_dict(torch.load(best_model_path))

        test_metrics = evaluate(test_model, test_loader, device, forces_rms)
        test_log = (
            f"TEST RESULTS - Loss={test_metrics['loss']:.4f}, "
            f"MAE_E={test_metrics['mae_e']:.2f} meV, RMSE_E={test_metrics['rmse_e']:.2f} meV, "
            f"MAE_F={test_metrics['mae_f']:.2f} meV/Å, RMSE_F={test_metrics['rmse_f']:.2f} meV/Å"
        )
        logger.info(test_log)
    else:
        logger.warning(f"Best model not found at {best_model_path}")

    logger.info(f"Training complete. Best Val Loss: {best_val_loss:.4f}")

if __name__ == '__main__':
    main()

from torch.utils.data import Dataset, DataLoader
from src.load_co3d import CO3DDatasetLoader
from typing import List, Tuple, Dict
import torchvision.transforms as T
from pathlib import Path
from tqdm import tqdm
import numpy as np
import random
import torch


class MultiViewPairDataset(Dataset):
    def __init__(self, 
                 loader,
                 split: str = 'train',
                 image_size: Tuple[int, int] = (256, 256),
                 max_angle_diff: float = 45.0,
                 min_angle_diff: float = 15.0,
                 max_pairs_per_sequence: int = 10):
        """
        Dataset dla par widoków.
        
        Args:
            loader: Instancja CO3DDatasetLoader
            split: 'train' lub 'val'
            image_size: Rozmiar obrazów wyjściowych
            max_angle_diff: Maksymalna różnica kątów między widokami (w stopniach)
            min_angle_diff: Minimalna różnica kątów między widokami (w stopniach)
            max_pairs_per_sequence: Maksymalna liczba par z jednej sekwencji
        """
        self.loader = loader
        self.image_size = image_size
        self.max_angle_diff = np.radians(max_angle_diff)
        self.min_angle_diff = np.radians(min_angle_diff)
        
        print(f"Initializing {split} dataset...")
        
        # Podstawowa transformacja obrazów
        self.transform = T.Compose([
            T.Resize(image_size),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        # Wybierz sekwencje z dobrą rekonstrukcją 3D
        with tqdm(desc="Filtering sequences", unit="seq") as pbar:
            self.sequences = self._filter_sequences(pbar)
        
        # Podziel na train/val (80/20)
        random.seed(42)  # dla powtarzalności
        random.shuffle(self.sequences)
        split_idx = int(len(self.sequences) * 0.8)
        self.sequences = self.sequences[:split_idx] if split == 'train' else self.sequences[split_idx:]
        
        # Przygotuj pary widoków
        print(f"Preparing view pairs for {len(self.sequences)} sequences...")
        self.view_pairs = self._prepare_view_pairs(max_pairs_per_sequence)
        print(f"Created {len(self.view_pairs)} view pairs")


    def _filter_sequences(self, pbar) -> List[str]:
        """Wybiera sekwencje z dobrą rekonstrukcją 3D"""
        good_sequences = []
        for seq in self.loader.sequence_annotations:
            seq_name = seq['sequence_name']
            # Sprawdź czy istnieje pointcloud i czy jakość widoków jest dobra
            if (Path(self.loader.base_path) / seq_name / 'pointcloud.ply').exists() and \
               seq['viewpoint_quality_score'] > 0.5:  # przykładowy próg
                good_sequences.append(seq_name)
            pbar.update(1)
        return good_sequences
    

    def _compute_view_angle(self, R1: np.ndarray, R2: np.ndarray) -> float:
        """Oblicza kąt między dwoma orientacjami kamery"""
        # Wektor "do przodu" dla każdej kamery (oś Z)
        forward1 = R1[:, 2]
        forward2 = R2[:, 2]
        # Kąt między wektorami
        cos_angle = np.clip(np.dot(forward1, forward2), -1.0, 1.0)
        return np.arccos(cos_angle)
    

    def _prepare_view_pairs(self, max_pairs_per_sequence: int) -> List[Dict]:
        """Przygotowuje pary widoków dla wszystkich sekwencji"""
        view_pairs = []
        
        for sequence_id in tqdm(self.sequences, desc="Preparing view pairs"):
            sequence_data = self.loader.get_sequence_data(sequence_id)
            frames = sequence_data['frames']
            
            # Znajdź pary widoków z odpowiednią różnicą kątów
            valid_pairs = []
            for i, frame1 in enumerate(frames):
                R1 = np.array(frame1['viewpoint']['R'])
                for j, frame2 in enumerate(frames[i+1:], i+1):
                    R2 = np.array(frame2['viewpoint']['R'])
                    angle_diff = self._compute_view_angle(R1, R2)
                    
                    if self.min_angle_diff <= angle_diff <= self.max_angle_diff:
                        valid_pairs.append({
                            'sequence_id': sequence_id,
                            'source_frame': frame1,
                            'target_frame': frame2,
                            'angle_diff': angle_diff
                        })
            
            # Wybierz losowo max_pairs_per_sequence par
            if valid_pairs:
                selected_pairs = random.sample(valid_pairs, 
                                            min(max_pairs_per_sequence, len(valid_pairs)))
                view_pairs.extend(selected_pairs)
        
        return view_pairs
    

    def __len__(self) -> int:
        return len(self.view_pairs)
    

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        pair = self.view_pairs[idx]
        sequence_id = pair['sequence_id']
        
        # Wczytaj dane źródłowego i docelowego widoku
        source_data = self.loader.load_frame_data(pair['source_frame'])
        target_data = self.loader.load_frame_data(pair['target_frame'])
        
        # Wczytaj chmurę punktów
        sequence_data = self.loader.get_sequence_data(sequence_id)
        points_3d = torch.tensor(sequence_data['points_3d']).float()
        
        # Przygotuj macierze K dla source i target
        def create_K_matrix(data):
            f = float(data['focal_length'][0] if isinstance(data['focal_length'], (list, tuple)) 
                     else data['focal_length'])
            px, py = map(float, data['principal_point'])
            return torch.tensor([
                [f, 0, px],
                [0, f, py],
                [0, 0, 1]
            ]).float()
        
        # Przygotuj dane wyjściowe z poprawioną strukturą
        return {
            'source_image': self.transform(source_data['image']),
            'target_image': self.transform(target_data['image']),
            'source_camera': {
                'R': torch.tensor(source_data['R']).float(),
                'T': torch.tensor(source_data['T']).float(),
                'K': create_K_matrix(source_data)
            },
            'target_camera': {
                'R': torch.tensor(target_data['R']).float(),
                'T': torch.tensor(target_data['T']).float(),
                'K': create_K_matrix(target_data)
            },
            'points_3d': points_3d,
            'sequence_id': sequence_id,
            'angle_diff': torch.tensor(pair['angle_diff']).float()
        }


def custom_collate(batch):
    """
    Custom collate function to handle variable-sized point clouds
    """
    # Separate the batch into individual components
    batch_dict = {
        'source_image': [],
        'target_image': [],
        'source_camera': {
            'R': [],
            'T': [],
            'K': []
        },
        'target_camera': {
            'R': [],
            'T': [],
            'K': []
        },
        'points_3d': [],  # This will be a list of differently sized point clouds
        'sequence_id': [],
        'angle_diff': []
    }

    # Collect items
    for item in batch:
        batch_dict['source_image'].append(item['source_image'])
        batch_dict['target_image'].append(item['target_image'])
        batch_dict['source_camera']['R'].append(item['source_camera']['R'])
        batch_dict['source_camera']['T'].append(item['source_camera']['T'])
        batch_dict['source_camera']['K'].append(item['source_camera']['K'])
        batch_dict['target_camera']['R'].append(item['target_camera']['R'])
        batch_dict['target_camera']['T'].append(item['target_camera']['T'])
        batch_dict['target_camera']['K'].append(item['target_camera']['K'])
        batch_dict['points_3d'].append(item['points_3d'])
        batch_dict['sequence_id'].append(item['sequence_id'])
        batch_dict['angle_diff'].append(item['angle_diff'])

    # Stack tensors where possible
    return {
        'source_image': torch.stack(batch_dict['source_image']),
        'target_image': torch.stack(batch_dict['target_image']),
        'source_camera': {
            'R': torch.stack(batch_dict['source_camera']['R']),
            'T': torch.stack(batch_dict['source_camera']['T']),
            'K': torch.stack(batch_dict['source_camera']['K'])
        },
        'target_camera': {
            'R': torch.stack(batch_dict['target_camera']['R']),
            'T': torch.stack(batch_dict['target_camera']['T']),
            'K': torch.stack(batch_dict['target_camera']['K'])
        },
        'points_3d': batch_dict['points_3d'],  # Keep as list of tensors
        'sequence_id': batch_dict['sequence_id'],
        'angle_diff': torch.stack(batch_dict['angle_diff'])
    }

def create_dataloaders(data_path: str,
                      batch_size: int = 8,
                      num_workers: int = 4,
                      **dataset_kwargs) -> Tuple[DataLoader, DataLoader]:
    """
    Tworzy DataLoadery dla zbiorów treningowego i walidacyjnego
    """
    loader = CO3DDatasetLoader(data_path)
    
    # Utwórz datasety
    train_dataset = MultiViewPairDataset(loader, split='train', **dataset_kwargs)
    val_dataset = MultiViewPairDataset(loader, split='val', **dataset_kwargs)
    
    # Utwórz data loadery z custom collate function
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=custom_collate
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=custom_collate
    )
    
    return train_loader, val_loader
from typing import Dict, Tuple
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import open3d as o3d
import numpy as np
import torch
import json
import gzip
import os

class CO3DDatasetLoader:
    def __init__(self, base_path: str):
        """
        Inicjalizacja loadera dla datasetu CO3D
        Args:
            base_path: Ścieżka do folderu kategorii (np. 'apple')
        """
        self.base_path = Path(base_path)
        self.frame_annotations = self._load_gzip_json('frame_annotations.jgz')
        self.sequence_annotations = self._load_gzip_json('sequence_annotations.jgz')
        
    def _load_gzip_json(self, filename: str) -> Dict:
        """Ładuje i dekoduje skompresowany plik JSON"""
        filepath = self.base_path / filename
        with gzip.open(filepath, 'r') as f:
            return json.loads(f.read().decode('utf-8'))
    
    def get_sequence_data(self, sequence_id: str) -> Dict:
        """
        Pobiera dane dla konkretnej sekwencji
        Args:
            sequence_id: ID sekwencji (np. '103_12148_21353')
        Returns:
            Dict zawierający informacje o sekwencji
        """
        # Znajdź wszystkie klatki dla danej sekwencji
        sequence_frames = [
            frame for frame in self.frame_annotations
            if frame['sequence_name'] == sequence_id
        ]
        
        # Pobierz odpowiadające pointcloudy
        pointcloud_path = self.base_path / sequence_id / 'pointcloud.ply'
        if pointcloud_path.exists():
            pcd = o3d.io.read_point_cloud(str(pointcloud_path))
            points_3d = np.asarray(pcd.points)
        else:
            points_3d = None
            
        sequence_data = {
            'frames': sequence_frames,
            'points_3d': points_3d,
            'sequence_info': next(
                seq for seq in self.sequence_annotations
                if seq['sequence_name'] == sequence_id
            )
        }
        
        return sequence_data
    
    def load_frame_data(self, frame_annotation: Dict) -> Dict:
        """
        Ładuje dane dla pojedynczej klatki
        Args:
            frame_annotation: Anotacja klatki z frame_annotations
        Returns:
            Dict zawierający dane klatki
        """
        sequence_name = frame_annotation['sequence_name']
        
        # Get image directory and list all image files
        image_dir = self.base_path / sequence_name / 'images'
        image_files = sorted(image_dir.glob('*.jpg'))  # Sort files naturally
        
        # Find the frame number from the index in sorted files
        frame_number = frame_annotation['frame_number']
        if frame_number >= len(image_files):
            raise ValueError(f"Frame number {frame_number} out of range for sequence {sequence_name}")
        
        image_path = image_files[frame_number]
        frame_basename = image_path.stem
        
        # Get corresponding mask and depth paths
        mask_path = self.base_path / sequence_name / 'masks' / f"{frame_basename}.png"
        depth_path = self.base_path / sequence_name / 'depth_masks' / f"{frame_basename}.png"
        
        # Ładowanie danych
        image = Image.open(image_path)
        
        # Get camera parameters from the viewpoint
        viewpoint = frame_annotation['viewpoint']
        
        frame_data = {
            'image': image,
            'mask': Image.open(mask_path) if mask_path.exists() else None,
            'depth': np.array(Image.open(depth_path)) if depth_path.exists() else None,
            'R': np.array(viewpoint['R']),
            'T': np.array(viewpoint['T']),
            # 'K': np.array(viewpoint['K']),
            'focal_length': viewpoint['focal_length'],
            'principal_point': viewpoint['principal_point'],
        }
        
        return frame_data

    def prepare_training_data(self, 
                            sequence_id: str,
                            image_size: Tuple[int, int] = (256, 256)) -> Dict[str, torch.Tensor]:
        """
        Przygotowuje dane do treningu dla pojedynczej sekwencji
        Args:
            sequence_id: ID sekwencji
            image_size: Docelowy rozmiar obrazów
        Returns:
            Dict z przetworzonymi danymi
        """
        sequence_data = self.get_sequence_data(sequence_id)
        frames = sequence_data['frames']
        points_3d = sequence_data['points_3d']
        
        # Przygotuj listy na dane
        processed_data = {
            'images': [],
            'masks': [],
            'camera_matrices': [],
            'points': []
        }
        
        # Przetwórz każdą klatkę
        for frame in tqdm(frames, desc=f"Processing sequence {sequence_id}"):
            frame_data = self.load_frame_data(frame)
            
            # Przeskaluj obraz
            image = frame_data['image'].resize(image_size, Image.BILINEAR)
            image = torch.tensor(np.array(image)) / 255.0
            
            # Przeskaluj maskę
            if frame_data['mask'] is not None:
                mask = frame_data['mask'].resize(image_size, Image.NEAREST)
                mask = torch.tensor(np.array(mask)) > 0
            else:
                mask = torch.ones(image_size)
            
            # Przygotuj macierz kamery
            R = torch.tensor(frame_data['R'])
            T = torch.tensor(frame_data['T'])
            f = frame_data['focal_length']
            px, py = frame_data['principal_point']
            
            # Stwórz macierz kamery
            K = torch.tensor([
                [f, 0, px],
                [0, f, py],
                [0, 0, 1]
            ])
            
            processed_data['images'].append(image)
            processed_data['masks'].append(mask)
            processed_data['camera_matrices'].append({
                'R': R,
                'T': T,
                'K': K
            })
            
        if points_3d is not None:
            processed_data['points'] = torch.tensor(points_3d)
            
        # Konwertuj listy na tensory
        processed_data['images'] = torch.stack(processed_data['images'])
        processed_data['masks'] = torch.stack(processed_data['masks'])
        
        return processed_data

def analyze_dataset(base_path: str):
    """
    Analizuje i wyświetla podstawowe informacje o datasecie
    """
    loader = CO3DDatasetLoader(base_path)
    
    # Zbierz statystyki
    total_sequences = len(loader.sequence_annotations)
    total_frames = len(loader.frame_annotations)
    
    sequences_with_pointcloud = 0
    for seq in loader.sequence_annotations:
        if (Path(base_path) / seq['sequence_name'] / 'pointcloud.ply').exists():
            sequences_with_pointcloud += 1
    
    print(f"Analiza datasetu w {base_path}:")
    print(f"Liczba sekwencji: {total_sequences}")
    print(f"Całkowita liczba klatek: {total_frames}")
    print(f"Sekwencje z pointcloud: {sequences_with_pointcloud}")
    
    # Przykładowa analiza pierwszej sekwencji
    first_sequence = loader.sequence_annotations[0]['sequence_name']
    sequence_data = loader.get_sequence_data(first_sequence)
    
    print(f"\nPrzykładowa sekwencja {first_sequence}:")
    print(f"Liczba klatek: {len(sequence_data['frames'])}")
    if sequence_data['points_3d'] is not None:
        print(f"Liczba punktów 3D: {len(sequence_data['points_3d'])}")

# Przykład użycia:
if __name__ == "__main__":
    # Podstawowa analiza
    analyze_dataset("path/to/apple")
    
    # Przygotowanie danych treningowych dla jednej sekwencji
    loader = CO3DDatasetLoader("path/to/apple")
    training_data = loader.prepare_training_data("103_12148_21353")
    
    print("\nPrzygotowane dane treningowe:")
    for key, value in training_data.items():
        if isinstance(value, torch.Tensor):
            print(f"{key}: shape = {value.shape}")
        elif isinstance(value, list):
            print(f"{key}: {len(value)} elements")
from datasets.load_co3d import CO3DDatasetLoader
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
import open3d as o3d
import numpy as np
import torch

class CO3DVisualizer:
    def __init__(self, loader):
        self.loader = loader

    def visualize_sequence_frames(self, sequence_id: str, num_frames: int = 5):
        """
        Wyświetla obrazy, maski i mapy głębokości dla wybranych klatek z sekwencji
        """
        sequence_data = self.loader.get_sequence_data(sequence_id)
        frames = sequence_data['frames']
        
        # Wybierz równomiernie rozłożone klatki
        step = len(frames) // num_frames
        selected_frames = frames[::step][:num_frames]
        
        fig, axes = plt.subplots(3, num_frames, figsize=(4*num_frames, 10))
        fig.suptitle(f'Sequence: {sequence_id}')
        
        for i, frame in enumerate(selected_frames):
            frame_data = self.loader.load_frame_data(frame)
            
            # Obraz
            axes[0, i].imshow(frame_data['image'])
            axes[0, i].set_title(f'Frame {frame["frame_number"]}')
            axes[0, i].axis('off')
            
            # Maska
            if frame_data['mask'] is not None:
                axes[1, i].imshow(frame_data['mask'], cmap='gray')
                axes[1, i].set_title('Mask')
                axes[1, i].axis('off')
            
            # Mapa głębokości
            if frame_data['depth'] is not None:
                axes[2, i].imshow(frame_data['depth'], cmap='viridis')
                axes[2, i].set_title('Depth')
                axes[2, i].axis('off')
        
        plt.tight_layout()
        plt.show()

    def visualize_cameras_and_pointcloud(self, sequence_id: str):
        """
        Wizualizuje chmurę punktów i pozycje kamer w 3D używając plotly
        """
        sequence_data = self.loader.get_sequence_data(sequence_id)
        points_3d = sequence_data['points_3d']
        frames = sequence_data['frames']

        # Przygotuj dane do wizualizacji
        camera_positions = []
        for frame in frames:
            R = np.array(frame['viewpoint']['R'])
            T = np.array(frame['viewpoint']['T'])
            camera_positions.append(-np.dot(R.T, T))  # Pozycja kamery w świecie

        camera_positions = np.array(camera_positions)

        # Stwórz wizualizację plotly
        fig = go.Figure()

        # Dodaj chmurę punktów
        if points_3d is not None:
            fig.add_trace(go.Scatter3d(
                x=points_3d[:, 0],
                y=points_3d[:, 1],
                z=points_3d[:, 2],
                mode='markers',
                marker=dict(
                    size=1,
                    color='blue',
                    opacity=0.8
                ),
                name='3D Points'
            ))

        # Dodaj pozycje kamer
        fig.add_trace(go.Scatter3d(
            x=camera_positions[:, 0],
            y=camera_positions[:, 1],
            z=camera_positions[:, 2],
            mode='markers',
            marker=dict(
                size=5,
                color='red',
                symbol='square'
            ),
            name='Cameras'
        ))

        # Ustaw parametry wizualizacji
        fig.update_layout(
            scene=dict(
                aspectmode='data',
                camera=dict(
                    up=dict(x=0, y=1, z=0)
                )
            ),
            title=f'Sequence: {sequence_id} - 3D Visualization'
        )

        fig.show()

    def visualize_frame_with_projection(self, sequence_id: str, frame_number: int):
        """
        Wizualizuje pojedynczą klatkę z projekcją punktów 3D
        """
        sequence_data = self.loader.get_sequence_data(sequence_id)
        frame = next(f for f in sequence_data['frames'] if f['frame_number'] == frame_number)
        frame_data = self.loader.load_frame_data(frame)
        points_3d = sequence_data['points_3d']

        if points_3d is None:
            print("No 3D points available for this sequence")
            return

        # Przekształć punkty do układu kamery
        R = frame_data['R']
        T = frame_data['T']
        focal_length = frame_data['focal_length']
        # Handle focal length which might be a tuple/list
        fx = float(focal_length[0] if isinstance(focal_length, (list, tuple)) else focal_length)
        
        principal_point = frame_data['principal_point']
        px, py = float(principal_point[0]), float(principal_point[1])

        # Projekcja punktów
        points_cam = np.dot(points_3d, R.T) + T
        points_2d = np.zeros((points_cam.shape[0], 2))
        
        # Use broadcasting with scalar values
        points_2d[:, 0] = fx * points_cam[:, 0] / points_cam[:, 2] + px
        points_2d[:, 1] = fx * points_cam[:, 1] / points_cam[:, 2] + py

        # Wizualizacja
        plt.figure(figsize=(12, 8))
        plt.imshow(frame_data['image'])
        plt.plot(points_2d[:, 0], points_2d[:, 1], 'r.', markersize=1, alpha=0.5)
        plt.title(f'Frame {frame_number} with projected 3D points')
        plt.axis('off')
        plt.show()

# Przykład użycia:
if __name__ == "__main__":
    loader = CO3DDatasetLoader("path/to/apple")
    visualizer = CO3DVisualizer(loader)
    
    # Wybierz pierwszą sekwencję
    sequence_id = loader.sequence_annotations[0]['sequence_name']
    
    # Wizualizuj klatki
    visualizer.visualize_sequence_frames(sequence_id)
    
    # Wizualizuj chmurę punktów i kamery
    visualizer.visualize_cameras_and_pointcloud(sequence_id)
    
    # Wizualizuj pojedynczą klatkę z projekcją
    visualizer.visualize_frame_with_projection(sequence_id, 0)
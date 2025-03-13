import os
import torch
import random
import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class DummyDataset(Dataset):
    def __init__(
        self,
        base_folder: str,
        num_samples=100000,
        width=1024,
        height=576,
        sample_frames=25,
    ):
        """
        Args:
            num_samples (int): Number of samples in the dataset.
            channels (int): Number of channels, default is 3 for RGB.
        """
        self.num_samples = num_samples
        # Define the path to the folder containing video frames
        self.base_folder = base_folder
        self.folders = os.listdir(self.base_folder)
        self.channels = 3
        self.width = width
        self.height = height
        self.sample_frames = sample_frames

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        """
        Args:
            idx (int): Index of the sample to return.

        Returns:
            dict: A dictionary containing the 'pixel_values' tensor of shape (16, channels, 320, 512).
        """
        # Randomly select a folder (representing a video) from the base folder
        chosen_folder = random.choice(self.folders)
        folder_path = os.path.join(self.base_folder, chosen_folder)
        frames = os.listdir(folder_path)
        # Sort the frames by name
        frames.sort()

        # Ensure the selected folder has at least `sample_frames`` frames
        if len(frames) < self.sample_frames:
            raise ValueError(
                f"The selected folder '{chosen_folder}' contains fewer than `{self.sample_frames}` frames."
            )

        # Randomly select a start index for frame sequence
        start_idx = random.randint(0, len(frames) - self.sample_frames)
        selected_frames = frames[start_idx : start_idx + self.sample_frames]

        # Initialize a tensor to store the pixel values
        pixel_values = torch.empty(
            (self.sample_frames, self.channels, self.height, self.width)
        )

        # Load and process each frame
        for i, frame_name in enumerate(selected_frames):
            frame_path = os.path.join(folder_path, frame_name)
            with Image.open(frame_path) as img:
                # Resize the image and convert it to a tensor
                img_resized = img.resize((self.width, self.height))
                img_tensor = torch.from_numpy(np.array(img_resized)).float()

                # Normalize the image by scaling pixel values to [-1, 1]
                img_normalized = img_tensor / 127.5 - 1

                # Rearrange channels if necessary
                if self.channels == 3:
                    img_normalized = img_normalized.permute(2, 0, 1)  # For RGB images
                elif self.channels == 1:
                    img_normalized = img_normalized.mean(
                        dim=2, keepdim=True
                    )  # For grayscale images

                pixel_values[i] = img_normalized
        return {"pixel_values": pixel_values}


class DummyDataset2(Dataset):
    def __init__(
        self,
        base_folder: str,
        num_samples=100000,
        width=1024,
        height=576,
        sample_frames=25,
    ):
        """
        Args:
            num_samples (int): Number of samples in the dataset.
            width (int): Width of the frames.
            height (int): Height of the frames.
            sample_frames (int): Number of frames to sample per video.
        """
        self.num_samples = num_samples
        self.base_folder = base_folder
        self.folders = os.listdir(self.base_folder)
        self.channels = 3
        self.width = width
        self.height = height
        self.sample_frames = sample_frames
        self.frame_skip = 3  # Prende un frame ogni 3 o 4 per simulare 7 FPS

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        """
        Args:
            idx (int): Index of the sample to return.

        Returns:
            dict: A dictionary containing the 'pixel_values' tensor of shape (sample_frames, channels, height, width).
        """
        while True:  # Continua a cercare una sequenza valida
            # Randomly select a folder (representing a video) from the base folder
            chosen_folder = random.choice(self.folders)
            folder_path = os.path.join(self.base_folder, chosen_folder)
            frames = sorted(os.listdir(folder_path))

            # Verifica se ci sono abbastanza frame per il campionamento
            min_required_frames = self.sample_frames * self.frame_skip
            if len(frames) < min_required_frames:
                continue  # Prova un altro video invece di lanciare un errore

            # Seleziona un indice casuale di partenza, assicurandosi che ci siano abbastanza frame rimanenti
            start_idx = random.randint(0, len(frames) - min_required_frames)

            # Seleziona i frame a intervalli di 3-4 per simulare 7 FPS
            selected_frames = frames[
                start_idx : start_idx + min_required_frames : self.frame_skip
            ]

            # Assicura che esattamente `sample_frames` vengano presi
            if len(selected_frames) == self.sample_frames:
                break  # Esce dal loop solo quando trova una sequenza valida

        # Inizializza il tensor per i frame selezionati
        pixel_values = torch.empty(
            (self.sample_frames, self.channels, self.height, self.width)
        )

        for i, frame_name in enumerate(selected_frames):
            frame_path = os.path.join(folder_path, frame_name)
            with Image.open(frame_path) as img:
                img_resized = img.resize((self.width, self.height))
                img_tensor = torch.from_numpy(np.array(img_resized)).float()
                img_normalized = img_tensor / 127.5 - 1  # Normalizzazione [-1, 1]

                # Riordina i canali per PyTorch (C, H, W)
                if self.channels == 3:
                    img_normalized = img_normalized.permute(2, 0, 1)
                elif self.channels == 1:
                    img_normalized = img_normalized.mean(dim=2, keepdim=True)

                pixel_values[i] = img_normalized

        return {"pixel_values": pixel_values}


class DummyDataset3(Dataset):
    def __init__(
        self,
        base_folder: str,
        num_samples=100000,
        width=1024,
        height=576,
        sample_frames=10,
    ):
        """
        Args:
            num_samples (int): Number of samples in the dataset.
            width (int): Width of the frames.
            height (int): Height of the frames.
            sample_frames (int): Number of frames to sample per video.
        """
        self.num_samples = num_samples
        self.base_folder = base_folder
        self.folders = os.listdir(self.base_folder)
        self.channels = 3
        self.width = width
        self.height = height
        self.sample_frames = sample_frames

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        """
        Args:
            idx (int): Index of the sample to return.

        Returns:
            dict: A dictionary containing the 'pixel_values' tensor of shape (sample_frames, channels, height, width).
        """
        while True:  # Continua a cercare una sequenza valida
            # Seleziona una cartella video casuale
            chosen_folder = random.choice(self.folders)
            folder_path = os.path.join(self.base_folder, chosen_folder)
            frames = sorted(os.listdir(folder_path))

            if len(frames) < self.sample_frames:
                continue  # Prova un altro video invece di lanciare un errore

            # Usa il primo frame come riferimento
            ref_frame_path = os.path.join(folder_path, frames[0])

            # Seleziona i successivi `sample_frames - 1` come continua
            selected_frames = [frames[0]] + frames[1 : self.sample_frames]
            if len(selected_frames) == self.sample_frames:
                break  # Esce dal loop solo quando trova una sequenza valida

        # Inizializza il tensor per i frame selezionati
        pixel_values = torch.empty(
            (self.sample_frames, self.channels, self.height, self.width)
        )

        for i, frame_name in enumerate(selected_frames):
            frame_path = os.path.join(folder_path, frame_name)
            with Image.open(frame_path) as img:
                img_resized = img.resize((self.width, self.height))
                img_tensor = torch.from_numpy(np.array(img_resized)).float()
                img_normalized = img_tensor / 127.5 - 1  # Normalizzazione [-1, 1]

                # Riordina i canali per PyTorch (C, H, W)
                if self.channels == 3:
                    img_normalized = img_normalized.permute(2, 0, 1)
                elif self.channels == 1:
                    img_normalized = img_normalized.mean(dim=2, keepdim=True)

                pixel_values[i] = img_normalized

        return {"pixel_values": pixel_values}

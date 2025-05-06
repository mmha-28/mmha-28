import os
import os
import io
import random
import numpy as np
from numpy.lib.function_base import disp
import torch
from torchvision import transforms
import warnings
from decord import VideoReader, cpu
from torch.utils.data import Dataset
from .random_erasing import RandomErasing
from .video_transforms import (
    Compose, Resize, CenterCrop, Normalize,
    create_random_augment, random_short_side_scale_jitter, 
    random_crop,random_resized_crop_event, random_resized_crop_with_shift, random_resized_crop,
    horizontal_flip, random_short_side_scale_jitter, uniform_crop, 
)
from .volume_transforms import ClipToTensor, ClipToTensor_event
from PIL import Image

try:
    from petrel_client.client import Client
    has_client = True
except ImportError:
    has_client = False

class VideoClsDataset_sparse(Dataset):
    """Load your own video classification dataset."""

    def __init__(self, anno_path, prefix='', split=' ', mode='train', clip_len=8,
                 frame_sample_rate=2, crop_size=224, short_side_size=256,
                 new_height=256, new_width=340, keep_aspect_ratio=True,
                 num_segment=1, num_crop=1, test_num_segment=10, test_num_crop=3,
                 args=None):
        self.anno_path = anno_path
        self.prefix = prefix
        self.split = split
        self.mode = mode
        self.clip_len = clip_len
        self.frame_sample_rate = frame_sample_rate
        self.crop_size = crop_size
        self.short_side_size = short_side_size
        self.new_height = new_height
        self.new_width = new_width
        self.keep_aspect_ratio = keep_aspect_ratio
        self.num_segment = num_segment
        self.test_num_segment = test_num_segment
        self.num_crop = num_crop
        self.test_num_crop = test_num_crop
        self.args = args
        self.aug = False
        self.rand_erase = False
        assert num_segment == 1
        if self.mode in ['train']:
            self.aug = True
            if self.args.reprob > 0:
                self.rand_erase = True
        if VideoReader is None:
            raise ImportError("Unable to import `decord` which is required to read videos.")

        import pandas as pd
        cleaned = pd.read_csv(self.anno_path, header=None, delimiter=self.split)
        folder_paths = list(cleaned.values[:, 0])
        labels= list(cleaned.values[:, 1])
        all_frames_paths = []
        all_labels = []       

        for folder, label in zip(folder_paths, labels):
            frame_files = sorted(
            [os.path.join(folder, f) for f in os.listdir(folder) 
             if f.endswith(('.jpg', '.png','.npy'))]
            )

        



            # Sample frames based on `num_frames` and `frame_sample_rate`
            for start in range(0, len(frame_files), self.clip_len * frame_sample_rate):
                pos_sample=frame_files[start : start + self.clip_len * frame_sample_rate : frame_sample_rate]
                    
                if len(pos_sample) < self.clip_len // 2:

                    continue  # Skip this sequence
                if len(pos_sample) < self.clip_len:
                    indices = np.linspace(0, len(pos_sample) - 1, self.clip_len).astype(int)
                    pos_sample = [pos_sample[i] for i in indices]
                all_frames_paths.append(pos_sample)
                all_labels.append(label)
            
 

        self.client = None
        if has_client:
            self.client = Client('~/petreloss.conf')
        self.dataset_samples=all_frames_paths
        self.label_array=all_labels


        if (mode == 'train'):
            pass

        elif (mode == 'validation'):
            self.data_transform = Compose([
                Resize(self.short_side_size, interpolation='bilinear'),
                CenterCrop(size=(self.crop_size, self.crop_size)),
                ClipToTensor(),
                Normalize(mean=[0.485, 0.456, 0.406],
                                           std=[0.229, 0.224, 0.225])
            ]) 
            self.data_transform_event = Compose([
                Resize(self.short_side_size, interpolation='nearest'),
                CenterCrop(size=(self.crop_size, self.crop_size)),
                ClipToTensor_event()])        
            self.data_transform_thermal = Compose([
                Resize(self.short_side_size, interpolation='bilinear'),
                CenterCrop(size=(self.crop_size, self.crop_size)),
                ClipToTensor(),
                Normalize(mean=[0.483, 0.027, 0.508],
                                           std=[0.133, 0.106, 0.194])
            ])    
            self.data_transform_depth = Compose([
                Resize(self.short_side_size, interpolation='bilinear'),
                CenterCrop(size=(self.crop_size, self.crop_size)),
                ClipToTensor(),
                Normalize(mean=[0.591, 0.278, 0.258],
                                           std=[0.463, 0.355, 0.324])
            ])          
        elif mode == 'test':
            self.data_resize = Compose([
                Resize(size=(short_side_size), interpolation='bilinear')
            ])
            self.data_resize_event = Compose([
                Resize(size=(short_side_size), interpolation='nearest')
            ])            
            self.data_transform = Compose([
                ClipToTensor(),
                Normalize(mean=[0.485, 0.456, 0.406],
                                           std=[0.229, 0.224, 0.225])
            ])
            self.data_transform_event = Compose([
                ClipToTensor_event()])        
            self.data_transform_thermal = Compose([
                ClipToTensor(),
                Normalize(mean=[0.483, 0.027, 0.508],
                                           std=[0.133, 0.106, 0.194])
            ])    
            self.data_transform_depth = Compose([
                ClipToTensor(),
                Normalize(mean=[0.591, 0.278, 0.258],
                                           std=[0.463, 0.355, 0.324])
            ])    
            self.test_seg = []
            self.test_dataset = []
            self.test_label_array = []
            for ck in range(self.test_num_segment):
                for cp in range(self.test_num_crop):
                    for idx in range(len(self.label_array)):
                        sample_label = self.label_array[idx]
                        self.test_label_array.append(sample_label)
                        self.test_dataset.append(self.dataset_samples[idx])
                        self.test_seg.append((ck, cp))


    def __getitem__(self, index):
        if self.mode == 'train':
            args = self.args 

            sample = self.dataset_samples[index]
            
            buffer,paths = load_images_from_folder(sample, self.clip_len, self.frame_sample_rate)
            
            if len(buffer) == 0:

                while len(buffer) == 0:
                    warnings.warn(f"Frames not found for {sample}")
                    index = np.random.randint(self.__len__())
                    sample = self.dataset_samples[index]
                    warnings.warn(f"here{sample}")
                    buffer,paths = load_images_from_folder(sample, self.clip_len, self.frame_sample_rate)
                    buffer = self._aug_frame(buffer,paths, args)
            if args.num_sample > 1:pass
            else:


                buffer = self._aug_frame(buffer,paths, args)
            if not np.isfinite(buffer).any():  # Check for NaN/Inf
                print(f"NaN or Inf found in buffer at index {index}")
                print(f"Buffer stats - min: {np.min(buffer)}, max: {np.max(buffer)}, mean: {np.mean(buffer)}")
                buffer.stop

            return buffer, self.label_array[index], index, {}

        elif self.mode == 'validation':
            sample = self.dataset_samples[index]
            buffer,paths = load_images_from_folder(sample, self.clip_len, self.frame_sample_rate)


            if len(buffer) == 0:
                while len(buffer) == 0:
                    warnings.warn("video {} not correctly loaded during validation".format(sample))
                    index = np.random.randint(self.__len__())
                    sample = self.dataset_samples[index]
                    buffer,paths = load_images_from_folder(sample, self.clip_len, self.frame_sample_rate)
            if "depth" in paths[0].lower():
                buffer = self.data_transform_depth(buffer)
            elif "thermal" in paths[0].lower():
                buffer = self.data_transform_thermal(buffer)
            elif paths[0].endswith(".npy"):
                buffer = self.data_transform_event(buffer)
            else:
                buffer = self.data_transform(buffer)
            return buffer, self.label_array[index], {"path":sample}

        elif self.mode == 'test':
            sample = self.test_dataset[index]
            chunk_nb, split_nb = self.test_seg[index]
            buffer,paths = load_images_from_folder(sample, self.clip_len, self.frame_sample_rate)

            while len(buffer) == 0:
                warnings.warn("video {}, temporal {}, spatial {} not found during testing".format(\
                    str(self.test_dataset[index]), chunk_nb, split_nb))
                index = np.random.randint(self.__len__())
                sample = self.test_dataset[index]
                chunk_nb, split_nb = self.test_seg[index]
                buffer = load_images_from_folder(sample, self.clip_len, self.frame_sample_rate)
            if paths[0].endswith(".npy"):
                buffer = self.data_resize_event(buffer)
            else:
                buffer = self.data_resize(buffer)
            if isinstance(buffer, list):
                buffer = np.stack(buffer, 0)
            if self.test_num_crop == 1:
                spatial_step = 1.0 * (max(buffer.shape[1], buffer.shape[2]) - self.short_side_size) / 2
                spatial_start = int(spatial_step)
            else:
                spatial_step = 1.0 * (max(buffer.shape[1], buffer.shape[2]) - self.short_side_size) \
                                    / (self.test_num_crop - 1)
                spatial_start = int(split_nb * spatial_step)
            if buffer.shape[1] >= buffer.shape[2]:
                buffer = buffer[:, spatial_start:spatial_start + self.short_side_size, :, :]
            else:
                buffer = buffer[:, :, spatial_start:spatial_start + self.short_side_size, :]

            if "depth" in paths[0].lower():
                buffer = self.data_transform_depth(buffer)
            elif "thermal" in paths[0].lower():
                buffer = self.data_transform_thermal(buffer)
            elif paths[0].endswith(".npy"):
                buffer = self.data_transform_event(buffer)
            else:
                buffer = self.data_transform(buffer)

            return buffer, self.test_label_array[index], {"path":sample}
        else:
            raise NameError('mode {} unkown'.format(self.mode))

    def _aug_frame(
        self,
        buffer, paths,
        args,
    ):

        aug_transform = create_random_augment(
            input_size=(self.crop_size, self.crop_size),
            auto_augment=args.aa,
            interpolation=args.train_interpolation,
        )


        normalized_frames = []
        
        for frame, path in zip(buffer, paths):

                # If the file is an event stream (.npy), skip augmentation
            if path.endswith(".npy"):
                tensor_frame = torch.tensor(frame)  # Convert to tensor without modifications
                tensor_frame = tensor_frame.permute(2, 0, 1)
            else:
                    # Convert image to uint8 before passing to PIL
                img = Image.fromarray(frame.astype(np.uint8))


                    # Apply augmentations
                img = aug_transform([img])[0]

                    # Convert to tensor
                tensor_frame = transforms.ToTensor()(img)
            tensor_frame = tensor_frame.permute(1, 2, 0)
                # Normalize based on frame type
            if "depth" in path.lower():
                mean = torch.tensor([0.591, 0.278, 0.258], dtype=tensor_frame.dtype, device=tensor_frame.device)
                std = torch.tensor([0.463, 0.355, 0.324], dtype=tensor_frame.dtype, device=tensor_frame.device)
                tensor_frame = tensor_normalize(tensor_frame, mean, std)
            elif "thermal" in path.lower():
                mean = torch.tensor([0.483, 0.027, 0.508], dtype=tensor_frame.dtype, device=tensor_frame.device)
                std = torch.tensor([0.133, 0.106, 0.194], dtype=tensor_frame.dtype, device=tensor_frame.device)
                tensor_frame = tensor_normalize(tensor_frame, mean, std)
            elif path.endswith(".npy"):  # Event-stream frames (skip normalization)
                pass
            else:
                tensor_frame = tensor_normalize(tensor_frame, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

            normalized_frames.append(tensor_frame)
        
        buffer = torch.stack(normalized_frames)



            # T H W C -> C T H W.
        buffer = buffer.permute(3, 0, 1, 2)
            # Perform data augmentation.
        scl, asp = (
                [0.08, 1.0],
                [0.75, 1.3333],
            )
        buffer = spatial_sampling(
                buffer,
                paths,
                spatial_idx=-1,
                min_scale=256,
                max_scale=320,
                crop_size=self.crop_size,
                random_horizontal_flip=False if args.data_set == 'SSV2' else True ,
                inverse_uniform_sampling=False,
                aspect_ratio=asp,
                scale=scl,
                motion_shift=False
            )

        if self.rand_erase:
            erase_transform = RandomErasing(
                    args.reprob,
                    mode=args.remode,
                    max_count=args.recount,
                    num_splits=args.recount,
                    device="cpu",
                )
            buffer = buffer.permute(1, 0, 2, 3)
            buffer = erase_transform(buffer)
            buffer= buffer.permute(1, 0, 2, 3)
        return buffer

    def _get_seq_frames(self, video_size, num_frames, clip_idx=-1):
        seg_size = max(0., float(video_size - 1) / num_frames)
        max_frame = int(video_size) - 1
        seq = []
        # index from 1, must add 1
        if clip_idx == -1:
            for i in range(num_frames):
                start = int(np.round(seg_size * i))
                end = int(np.round(seg_size * (i + 1)))
                idx = min(random.randint(start, end), max_frame)
                seq.append(idx)
        else:
            num_segment = 1
            duration = seg_size / (num_segment + 1)
            for i in range(num_frames):
                start = int(np.round(seg_size * i))
                frame_index = start + int(duration * (clip_idx + 1))
                idx = min(frame_index, max_frame)
                seq.append(idx)
        return seq

    def loadvideo_decord(self, sample, chunk_nb=0):
        """Load video content using Decord"""
        fname = sample
        fname = os.path.join(self.prefix, fname)

        try:
            if self.keep_aspect_ratio:
                if "s3://" in fname:
                    video_bytes = self.client.get(fname)
                    vr = VideoReader(io.BytesIO(video_bytes),
                                     num_threads=1,
                                     ctx=cpu(0))
                else:
                    vr = VideoReader(fname, num_threads=1, ctx=cpu(0))
            else:
                if "s3://" in fname:
                    video_bytes = self.client.get(fname)
                    vr = VideoReader(io.BytesIO(video_bytes),
                                     width=self.new_width,
                                     height=self.new_height,
                                     num_threads=1,
                                     ctx=cpu(0))
                else:
                    vr = VideoReader(fname, width=self.new_width, height=self.new_height,
                                    num_threads=1, ctx=cpu(0))

            all_index = self._get_seq_frames(len(vr), self.clip_len, clip_idx=chunk_nb)
            vr.seek(0)
            buffer = vr.get_batch(all_index).asnumpy()
            return buffer
        except:
            print("video cannot be loaded by decord: ", fname)
            return []

    def __len__(self):
        if self.mode != 'test':
            return len(self.dataset_samples)
        else:
            return len(self.test_dataset)

from PIL import Image
import glob

from PIL import Image
import numpy as np
import glob
import os

import glob
import os
import numpy as np
from PIL import Image


def load_images_from_folder(folder_path, num_frames, frame_sample_rate=1):
    """Load a sequence of frames from a folder, supporting RGB, depth, and event stream (.npy) frames.
       Returns both the images and their corresponding file paths.
    """
    

    frames = []
    for frame_path in folder_path:
        if frame_path.endswith(".npy"):  # Event stream frames
            img = np.load(frame_path)  # Load as NumPy array
            if img.ndim == 2:  # Single-channel event frame

                img = np.expand_dims(img, axis=-1)  # Add a channel dimension

                img = np.repeat(img, 3, axis=-1)

        else:
            if "depth" in frame_path.lower():
                img = Image.open(frame_path).convert("RGB")  # Load RGB (3 channels)
            else:
                img = Image.open(frame_path).convert("RGB")  # Load RGB (3 channels)
                img = np.array(img) 
                
            img = np.array(img)  # Convert to NumPy array
            
        frames.append(img)

    if not np.isfinite(np.array(frames)).any():  # Check for NaN/Inf
        print(f"Corrupted data at {folder_path}, skipping...")
        folder_path.stop()
    return np.array(frames),folder_path  # Return images and corresponding paths

 
    


def spatial_sampling(
    frames,
    paths,
    spatial_idx=-1,
    min_scale=256,
    max_scale=320,
    crop_size=224,
    random_horizontal_flip=True,
    inverse_uniform_sampling=False,
    aspect_ratio=None,
    scale=None,
    motion_shift=False,
):
    """
    Perform spatial sampling on the given video frames. If spatial_idx is
    -1, perform random scale, random crop, and random flip on the given
    frames. If spatial_idx is 0, 1, or 2, perform spatial uniform sampling
    with the given spatial_idx.
    Args:
        frames (tensor): frames of images sampled from the video. The
            dimension is `num frames` x `height` x `width` x `channel`.
        spatial_idx (int): if -1, perform random spatial sampling. If 0, 1,
            or 2, perform left, center, right crop if width is larger than
            height, and perform top, center, buttom crop if height is larger
            than width.
        min_scale (int): the minimal size of scaling.
        max_scale (int): the maximal size of scaling.
        crop_size (int): the size of height and width used to crop the
            frames.
        inverse_uniform_sampling (bool): if True, sample uniformly in
            [1 / max_scale, 1 / min_scale] and take a reciprocal to get the
            scale. If False, take a uniform sample from [min_scale,
            max_scale].
        aspect_ratio (list): Aspect ratio range for resizing.
        scale (list): Scale range for resizing.
        motion_shift (bool): Whether to apply motion shift for resizing.
    Returns:
        frames (tensor): spatially sampled frames.
    """
    assert spatial_idx in [-1, 0, 1, 2]
    if spatial_idx == -1:
       
        if aspect_ratio is None and scale is None:

            frames, _ = random_short_side_scale_jitter(
                images=frames,
                min_size=min_scale,
                max_size=max_scale,
                inverse_uniform_sampling=inverse_uniform_sampling,
            )
            frames, _ = random_crop(frames, crop_size)
        else:


            transform_func = (
                random_resized_crop_with_shift
                if motion_shift
                else random_resized_crop
            )
            transform_func_event = (
                random_resized_crop_with_shift
                if motion_shift
                else random_resized_crop_event
            )
            if paths[0].endswith(".npy"):

                frames = transform_func_event(
                images=frames,
                target_height=crop_size,
                target_width=crop_size,
                scale=scale,
                ratio=aspect_ratio,)
            else:
                frames = transform_func(
                images=frames,
                target_height=crop_size,
                target_width=crop_size,
                scale=scale,
                ratio=aspect_ratio,)

        if random_horizontal_flip:
            frames, _ = horizontal_flip(0.5, frames)


    else:
        # The testing is deterministic and no jitter should be performed.
        # min_scale, max_scale, and crop_size are expect to be the same.
        assert len({min_scale, max_scale, crop_size}) == 1
        frames, _ = random_short_side_scale_jitter(
            frames, min_scale, max_scale
        )
        frames, _ = uniform_crop(frames, crop_size, spatial_idx)
    
    return frames


def tensor_normalize(tensor, mean, std):
    """
    Normalize a given tensor by subtracting the mean and dividing the std.
    Args:
        tensor (tensor): tensor to normalize.
        mean (tensor or list): mean value to subtract.
        std (tensor or list): std to divide.
    """
    if tensor.dtype == torch.uint8:
        tensor = tensor.float()
        tensor = tensor / 255.0
    mean = torch.tensor(mean, dtype=tensor.dtype, device=tensor.device)
    std = torch.tensor(std, dtype=tensor.dtype, device=tensor.device)
    tensor = tensor - mean
    tensor = tensor / std
    return tensor







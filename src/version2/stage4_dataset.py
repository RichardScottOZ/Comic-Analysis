"""
Stage 4 Dataset Loader

Dataset for training Stage 4 sequence modeling.
Expects panel embeddings from Stage 3 and creates training samples
for ComicsPAP-inspired tasks and Text-Cloze.

Dataset format:
- Loads pre-computed panel embeddings from Stage 3
- Creates masked sequences for panel picking
- Generates candidate sets for discriminative tasks
- Supports multiple tasks simultaneously
"""

import os
import json
import torch
import numpy as np
import zarr
from torch.utils.data import Dataset
from typing import Dict, List, Optional, Tuple
import random


class Stage4SequenceDataset(Dataset):
    """
    Dataset for Stage 4 sequence modeling tasks.
    
    Expected directory structure:
        embeddings_dir/
            panel_embeddings.zarr  # From Stage 3
            metadata.json          # Book/page/panel metadata
        
    Training tasks:
    1. Panel Picking: Select correct panel for masked position
    2. Character Coherence: Maintain character consistency
    3. Visual/Text Closure: Predict narrative continuation
    4. Caption Relevance: Align panels with captions
    5. Text-Cloze: Predict missing text
    6. Reading Order: Sequence ordering
    """
    
    def __init__(self,
                 embeddings_path: str,
                 metadata_path: str,
                 min_panels: int = 3,
                 max_panels: int = 16,
                 task_weights: Optional[Dict[str, float]] = None,
                 num_candidates: int = 5):
        """
        Args:
            embeddings_path: Path to panel_embeddings.zarr
            metadata_path: Path to metadata.json
            min_panels: Minimum panels per sequence
            max_panels: Maximum panels per sequence
            task_weights: Weights for sampling different tasks
            num_candidates: Number of candidates for discriminative tasks
        """
        self.min_panels = min_panels
        self.max_panels = max_panels
        self.num_candidates = num_candidates
        
        # Load embeddings
        self.embeddings = zarr.open(embeddings_path, mode='r')
        
        # Load metadata
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
        
        # Default task weights (uniform if not provided)
        if task_weights is None:
            self.task_weights = {
                'panel_picking': 1.0,
                'character_coherence': 0.5,
                'visual_closure': 0.8,
                'text_closure': 0.8,
                'caption_relevance': 0.5,
                'text_cloze': 1.0,
                'reading_order': 0.7
            }
        else:
            self.task_weights = task_weights
        
        # Build index of valid sequences
        self.sequences = self._build_sequence_index()
        
        print(f"Stage 4 Dataset initialized:")
        print(f"  - Total sequences: {len(self.sequences)}")
        print(f"  - Panel range: [{min_panels}, {max_panels}]")
        print(f"  - Task weights: {self.task_weights}")
    
    def _build_sequence_index(self) -> List[Dict]:
        """
        Build index of valid panel sequences.
        
        Returns:
            List of sequence dictionaries with metadata
        """
        sequences = []
        
        for item in self.metadata:
            num_panels = item.get('num_panels', 0)
            
            # Filter by panel count
            if num_panels < self.min_panels or num_panels > self.max_panels:
                continue
            
            sequences.append({
                'book_id': item['book_id'],
                'page_name': item['page_name'],
                'num_panels': num_panels,
                'embedding_indices': item['embedding_indices']
            })
        
        return sequences
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    def _load_panel_embeddings(self, indices: List[int]) -> np.ndarray:
        """
        Load panel embeddings by indices.
        
        Args:
            indices: List of embedding indices
            
        Returns:
            (N, D) panel embeddings
        """
        embeddings = []
        for idx in indices:
            emb = self.embeddings[idx]
            embeddings.append(emb)
        
        return np.stack(embeddings, axis=0)
    
    def _create_panel_picking_sample(self, 
                                     panel_embeddings: np.ndarray,
                                     num_panels: int) -> Dict:
        """
        Create a panel picking task sample.
        
        Randomly mask one panel and create candidates including the correct one.
        """
        # Randomly select position to mask
        mask_pos = random.randint(0, num_panels - 1)
        
        # Get correct panel
        correct_panel = panel_embeddings[mask_pos]
        
        # Create candidate pool (correct + random negatives)
        candidates = [correct_panel]
        
        # Sample random negatives from other positions
        other_positions = [i for i in range(num_panels) if i != mask_pos]
        if len(other_positions) >= self.num_candidates - 1:
            neg_positions = random.sample(other_positions, self.num_candidates - 1)
        else:
            # If not enough panels, repeat some
            neg_positions = random.choices(other_positions, k=self.num_candidates - 1)
        
        for pos in neg_positions:
            candidates.append(panel_embeddings[pos])
        
        # Shuffle candidates and track correct position
        candidate_pairs = list(enumerate(candidates))
        random.shuffle(candidate_pairs)
        correct_idx = [i for i, (orig_idx, _) in enumerate(candidate_pairs) if orig_idx == 0][0]
        candidates = [emb for _, emb in candidate_pairs]
        
        # Create context (all panels except masked)
        context_panels = np.concatenate([
            panel_embeddings[:mask_pos],
            panel_embeddings[mask_pos+1:]
        ], axis=0)
        
        # Context embedding (mean of surrounding panels)
        context_embedding = context_panels.mean(axis=0)
        
        return {
            'mask_position': mask_pos,
            'context_embedding': context_embedding,
            'candidates': np.stack(candidates, axis=0),
            'correct_idx': correct_idx
        }
    
    def _create_closure_sample(self,
                               panel_embeddings: np.ndarray,
                               num_panels: int,
                               closure_type: str = 'visual') -> Dict:
        """
        Create visual or text closure task sample.
        
        Select a split point and predict continuation.
        """
        # Split point (need at least 2 panels before)
        split_pos = random.randint(2, num_panels - 2)
        
        # Preceding panels
        preceding = panel_embeddings[:split_pos]
        
        # Correct continuation
        correct_panel = panel_embeddings[split_pos]
        
        # Create candidates
        candidates = [correct_panel]
        
        # Sample negatives from later panels
        later_positions = list(range(split_pos + 1, num_panels))
        if len(later_positions) >= self.num_candidates - 1:
            neg_positions = random.sample(later_positions, self.num_candidates - 1)
        else:
            neg_positions = random.choices(later_positions, k=self.num_candidates - 1)
        
        for pos in neg_positions:
            candidates.append(panel_embeddings[pos])
        
        # Shuffle
        candidate_pairs = list(enumerate(candidates))
        random.shuffle(candidate_pairs)
        correct_idx = [i for i, (orig_idx, _) in enumerate(candidate_pairs) if orig_idx == 0][0]
        candidates = [emb for _, emb in candidate_pairs]
        
        return {
            'preceding_panels': preceding,
            'candidates': np.stack(candidates, axis=0),
            'correct_idx': correct_idx,
            'closure_type': closure_type
        }
    
    def _create_reading_order_sample(self,
                                     panel_embeddings: np.ndarray,
                                     num_panels: int) -> Dict:
        """
        Create reading order task sample.
        
        Shuffle panels and predict correct ordering.
        """
        # Create correct order
        correct_order = list(range(num_panels))
        
        # Shuffle panels
        shuffled_indices = correct_order.copy()
        random.shuffle(shuffled_indices)
        
        shuffled_panels = panel_embeddings[shuffled_indices]
        
        # Create adjacency matrix for correct order
        adj_matrix = np.zeros((num_panels, num_panels), dtype=np.int64)
        for i in range(num_panels - 1):
            adj_matrix[i, i + 1] = 1
        
        return {
            'shuffled_panels': shuffled_panels,
            'shuffled_indices': shuffled_indices,
            'correct_order': correct_order,
            'adjacency_matrix': adj_matrix
        }
    
    def __getitem__(self, idx: int) -> Dict:
        """
        Get a training sample with randomly selected task.
        
        Returns:
            Dictionary containing:
            - panel_embeddings: (N, D) panel embeddings
            - panel_mask: (N,) binary mask
            - task_type: str indicating task
            - task_data: task-specific data
        """
        sequence = self.sequences[idx]
        
        # Load panel embeddings
        panel_embeddings = self._load_panel_embeddings(
            sequence['embedding_indices']
        )
        num_panels = sequence['num_panels']
        
        # Randomly select task based on weights
        tasks = list(self.task_weights.keys())
        weights = [self.task_weights[t] for t in tasks]
        task_type = random.choices(tasks, weights=weights, k=1)[0]
        
        # Create task sample
        if task_type == 'panel_picking':
            task_data = self._create_panel_picking_sample(panel_embeddings, num_panels)
        elif task_type in ['visual_closure', 'text_closure']:
            closure_type = 'visual' if task_type == 'visual_closure' else 'text'
            task_data = self._create_closure_sample(panel_embeddings, num_panels, closure_type)
        elif task_type == 'reading_order':
            task_data = self._create_reading_order_sample(panel_embeddings, num_panels)
        else:
            # For other tasks, return basic data
            task_data = {}
        
        # Create panel mask
        panel_mask = np.ones(num_panels, dtype=np.float32)
        
        # Pad to max_panels if needed
        if num_panels < self.max_panels:
            pad_size = self.max_panels - num_panels
            
            # Pad embeddings
            pad_emb = np.zeros((pad_size, panel_embeddings.shape[1]), dtype=np.float32)
            panel_embeddings = np.concatenate([panel_embeddings, pad_emb], axis=0)
            
            # Pad mask
            pad_mask = np.zeros(pad_size, dtype=np.float32)
            panel_mask = np.concatenate([panel_mask, pad_mask], axis=0)
        
        return {
            'panel_embeddings': torch.from_numpy(panel_embeddings).float(),
            'panel_mask': torch.from_numpy(panel_mask).float(),
            'task_type': task_type,
            'task_data': task_data,
            'metadata': {
                'book_id': sequence['book_id'],
                'page_name': sequence['page_name'],
                'num_panels': num_panels
            }
        }


def collate_stage4(batch: List[Dict]) -> Dict:
    """
    Collate function for Stage 4 dataset.
    
    Groups samples by task type for efficient training.
    """
    # Group by task type
    task_groups = {}
    for sample in batch:
        task_type = sample['task_type']
        if task_type not in task_groups:
            task_groups[task_type] = []
        task_groups[task_type].append(sample)
    
    # For simplicity, process all samples together
    # In practice, you might want to process by task type
    
    panel_embeddings = torch.stack([s['panel_embeddings'] for s in batch])
    panel_masks = torch.stack([s['panel_mask'] for s in batch])
    
    task_types = [s['task_type'] for s in batch]
    task_data_list = [s['task_data'] for s in batch]
    metadata = [s['metadata'] for s in batch]
    
    return {
        'panel_embeddings': panel_embeddings,
        'panel_mask': panel_masks,
        'task_types': task_types,
        'task_data': task_data_list,
        'metadata': metadata
    }


if __name__ == "__main__":
    print("Stage 4 Sequence Dataset Loader")
    print("=" * 60)
    print("\nThis dataset:")
    print("1. Loads panel embeddings from Stage 3")
    print("2. Creates training samples for multiple tasks:")
    print("   - Panel Picking (ComicsPAP)")
    print("   - Character Coherence")
    print("   - Visual/Text Closure")
    print("   - Caption Relevance")
    print("   - Text-Cloze")
    print("   - Reading Order")
    print("3. Supports task-specific sampling weights")
    print("4. Handles variable-length sequences with padding")

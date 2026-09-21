from torch.utils.data import Dataset, DataLoader
from clip import tokenize
from PIL import Image
import json
import torch
import os
from easydict import EasyDict as edict
from collections import defaultdict


class VG_Relation(Dataset):
    def __init__(self, transform = None, args=None):
        self.transform = transform
        self.num_triples = args.num_triples if args else 3
        self.root_dir = "data/visual_genome_data/vg_image/"
        
        with open("data/visual_genome_relation_aug.json", "r") as f:
            self.dataset = json.load(f)
        self.all_relations = list()
        for item in self.dataset:
            item["image_path"] = os.path.join(self.root_dir, item["image_path"])
            self.all_relations.append(item["relation_name"])
    
    def __len__(self):
        return len(self.dataset)
    
    def _process_triples(self, triples):
        triple_strings = [" ".join(t) for t in triples][:self.num_triples]
        
        num_real_triples = len(triple_strings)
        mask = torch.zeros(self.num_triples, dtype=torch.bool)
        if num_real_triples > 0:
            mask[:num_real_triples] = True
        
        padded_strings = triple_strings + [""] * (self.num_triples - num_real_triples)
        tokenized = tokenize(padded_strings, truncate=True)
        
        return tokenized, mask

    def __getitem__(self, index):    
        rel = self.dataset[index]
        # step1 
        image = Image.open(rel["image_path"]).convert('RGB')    
        # Get the bounding box that contains the relation
        image = image.crop((rel["bbox_x"], rel["bbox_y"], rel["bbox_x"]+rel["bbox_w"], rel["bbox_y"]+rel["bbox_h"]))
        
        if self.transform is not None:
            image = self.transform(image)
        
        true_triples = rel['true_triples']
        false_triples = rel['false_triples']
        
        tokenized_true_triples, true_mask = self._process_triples(true_triples)
        tokenized_false_triples, false_mask = self._process_triples(false_triples)
        
        true_caption = tokenize(rel.get('true_caption', ''), truncate=True)
        false_caption = tokenize(rel.get('false_caption', ''), truncate=True)

        return {
            "image": image,
            "true_triples": tokenized_true_triples,
            "true_mask": true_mask,
            "false_triples": tokenized_false_triples,
            "false_mask": false_mask,
            "true_caption": true_caption.squeeze(0),
            "false_caption": false_caption.squeeze(0),
            "relation": rel["relation_name"]
        }


class VG_Attribution(Dataset):
    def __init__(self, data_path=None, transform=None, args=None):
        self.transform = transform
        self.num_triples = args.num_triples if args else 3
        if 'adjchange' in data_path:
            self.root_dir = 'data/coco_data/'
        else:
            self.root_dir = "data/visual_genome_data/vg_image/"
        self.data_path = data_path
        
        with open(self.data_path, "r") as f:
            self.dataset = json.load(f)
        for item in self.dataset:
            # handle cases where image_path is already absolute
            if not os.path.isabs(item["image_path"]):
                item["image_path"] = os.path.join(self.root_dir, item["image_path"])

        if 'adjchange' not in data_path:
            self.all_attributes = [f"{item['attributes'][0]}_{item['attributes'][1]}" for item in self.dataset]

    def __len__(self):
        return len(self.dataset)
    
    def _process_triples(self, triples):
        triple_strings = [" ".join(t) for t in triples][:self.num_triples]
        
        num_real_triples = len(triple_strings)
        mask = torch.zeros(self.num_triples, dtype=torch.bool)
        if num_real_triples > 0:
            mask[:num_real_triples] = True
        
        padded_strings = triple_strings + [""] * (self.num_triples - num_real_triples)
        tokenized = tokenize(padded_strings, truncate=True)
        
        return tokenized, mask

    def __getitem__(self, index):
        scene = self.dataset[index]
        # step1 
        image = Image.open(scene["image_path"]).convert('RGB')
        # Get the bounding box that contains the relation
        if self.root_dir != 'data/coco_data/' and "bbox_x" in scene:
            image = image.crop((scene["bbox_x"], scene["bbox_y"], scene["bbox_x"] + scene["bbox_w"], scene["bbox_y"] + scene["bbox_h"]))

        if self.transform is not None:
            image = self.transform(image)

        true_triples = scene['true_triples']
        false_triples = scene['false_triples']
        
        tokenized_true_triples, true_mask = self._process_triples(true_triples)
        tokenized_false_triples, false_mask = self._process_triples(false_triples)

        true_caption = tokenize(scene.get('true_caption', ''), truncate=True)
        false_caption = tokenize(scene.get('false_caption', ''), truncate=True)

        return {
            "image": image,
            "true_triples": tokenized_true_triples,
            "true_mask": true_mask,
            "false_triples": tokenized_false_triples,
            "false_mask": false_mask,
            "true_caption": true_caption.squeeze(0),
            "false_caption": false_caption.squeeze(0),
            "relation": "attribution"
        }
    
class VisminDataset(Dataset):
    def __init__(self, dataset=None,transform=None, args=None):
        self.transform = transform
        self.num_triples = args.num_triples if args else 3
        
        # Load the dataset from Hugging Face
        self.dataset = dataset
        
        # Create mappings for faster lookups using parallel processing
        self._build_mappings()
            
    def _build_mappings(self):
        # First, gather all image_ids in parallel
        def extract_ids(examples):
            image_ids = examples['image_id']
            source_ids = examples['source_image_id']
            return {'image_id': image_ids, 'source_image_id': source_ids}
        
        # Process in batches with multiple processes
        id_data = self.dataset.map(
            extract_ids,
            batched=True,
            num_proc=4,  # Adjust based on available CPU cores
            remove_columns=['image', 'caption', 'bounding_boxes', 'edit_instruction', 'category', 'triples']
        )
        
        # Now build the mappings
        self.id_to_index = {}
        self.source_to_targets = defaultdict(list)
        
        for idx, (image_id, source_id) in enumerate(zip(id_data['image_id'], id_data['source_image_id'])):
            self.id_to_index[image_id] = idx
            if source_id:  # If this is a derived image
                self.source_to_targets[source_id].append(idx)
        
        # Convert defaultdict to regular dict for cleaner serialization if needed
        self.source_to_targets = dict(self.source_to_targets)
    
    def __len__(self):
        return len(self.dataset)
    
    def _process_triples(self, triples):
        triple_strings = [" ".join(t) for t in triples][:self.num_triples]
        
        num_real_triples = len(triple_strings)
        mask = torch.zeros(self.num_triples, dtype=torch.bool)
        if num_real_triples > 0:
            mask[:num_real_triples] = True
        
        padded_strings = triple_strings + [""] * (self.num_triples - num_real_triples)
        tokenized = tokenize(padded_strings, truncate=True)
        
        return tokenized, mask
    
    def __getitem__(self, index):
        # Get the current sample
        sample = self.dataset[index]
        
        # This is the true image
        true_image = sample['image']
        if self.transform is not None:
            true_image = self.transform(true_image)
        
        # Get the triples for the true image
        true_triples = sample['triples']
        
        # Find the corresponding false image
        false_image = None
        false_triples = None
        false_sample = None
        
        # If this is a source image (source_image_id is empty)
        if not sample['source_image_id']:
            # Check if this image has any derived images
            if sample['image_id'] in self.source_to_targets and self.source_to_targets[sample['image_id']]:
                # Get the first derived image
                derived_idx = self.source_to_targets[sample['image_id']][0]
                false_sample = self.dataset[derived_idx]
                false_image = false_sample['image']
                if self.transform is not None:
                    false_image = self.transform(false_image)
                false_triples = false_sample['triples']
        else:
            # This is a derived image, get its source image
            source_idx = self.id_to_index.get(sample['source_image_id'])
            if source_idx is not None:
                false_sample = self.dataset[source_idx]
                false_image = false_sample['image']
                if self.transform is not None:
                    false_image = self.transform(false_image)
                false_triples = false_sample['triples']
        
        # If we couldn't find a matching pair, use the same image as both true and false
        # This is a fallback and should be rare
        if false_image is None:
            false_image = true_image
            false_triples = true_triples
            false_sample = sample
        
        # Process triples
        tokenized_true_triples, true_mask = self._process_triples(true_triples)
        tokenized_false_triples, false_mask = self._process_triples(false_triples)
        
        # Tokenize captions
        true_caption = tokenize(sample.get('caption', ''), truncate=True)
        false_caption = tokenize(false_sample.get('caption', ''), truncate=True)
        
        return {
            "image": true_image,
            # "false_image": false_image,
            "true_triples": tokenized_true_triples,
            "true_mask": true_mask,
            "false_triples": tokenized_false_triples,
            "false_mask": false_mask,
            "true_caption": true_caption.squeeze(0),
            "false_caption": false_caption.squeeze(0),
            "relation": "vismin"
        }

class SemanticCompositionDataset(Dataset):
    def __init__(self, hf_dataset, transform=None, num_triples=5):
        """
        Dataset class for semantic composition evaluation
        Args:
            hf_dataset_name: Name of the HuggingFace dataset
            transform: Image transformation function
            num_triples: Maximum number of triples to use
        """
        self.transform = transform
        self.num_triples = num_triples
        
        # Load HuggingFace dataset
        self.dataset = hf_dataset
        
    def __len__(self):
        return len(self.dataset)
    
    def _process_triples(self, triples):
        """Process triples similar to your existing implementation"""
        triple_strings = [" ".join(t) for t in triples][:self.num_triples]
        
        num_real_triples = len(triple_strings)
        mask = torch.zeros(self.num_triples, dtype=torch.bool)
        if num_real_triples > 0:
            mask[:num_real_triples] = True
        
        padded_strings = triple_strings + [""] * (self.num_triples - num_real_triples)
        tokenized = tokenize(padded_strings, truncate=True)
        
        return tokenized, mask
    
    def __getitem__(self, index):
        sample = self.dataset[index]
        
        # Extract image
        image = sample['image'].convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        
        # Extract captions and triples
        true_caption = sample['true_caption']
        false_caption = sample['false_caption']
        true_triples = sample['true_triples']
        false_triples = sample['false_triples']
        
        # Process triples
        tokenized_true_triples, true_mask = self._process_triples(true_triples)
        tokenized_false_triples, false_mask = self._process_triples(false_triples)
        
        # Tokenize captions
        true_caption_tokenized = tokenize(true_caption, truncate=True)
        false_caption_tokenized = tokenize(false_caption, truncate=True)
        
        return {
            "image": image,
            "true_triples": tokenized_true_triples,
            "true_mask": true_mask,
            "false_triples": tokenized_false_triples,
            "false_mask": false_mask,
            "true_caption": true_caption_tokenized.squeeze(0),
            "false_caption": false_caption_tokenized.squeeze(0),
        }

class VisMinBenchDataset(Dataset):
    def __init__(self, hf_dataset, transform=None, args=None):
        self.transform = transform
        self.num_triples = args.num_triples if args else 5
        self.dataset = hf_dataset
        
    def __len__(self):
        return len(self.dataset)
    
    def _process_triples(self, triples):
        triple_strings = [" ".join(t) for t in triples][:self.num_triples]
        
        num_real_triples = len(triple_strings)
        mask = torch.zeros(self.num_triples, dtype=torch.bool)
        if num_real_triples > 0:
            mask[:num_real_triples] = True
        
        padded_strings = triple_strings + [""] * (self.num_triples - num_real_triples)
        tokenized = tokenize(padded_strings, truncate=True)
        
        return tokenized, mask

    def __getitem__(self, index):
        scene = self.dataset[index]
        
        # Get image
        image = scene['image'].convert('RGB')
        if self.transform is not None:
            image = self.transform(image)

        # Get triples
        true_triples = scene['true_triples']
        false_triples = scene['false_triples']
        
        tokenized_true_triples, true_mask = self._process_triples(true_triples)
        tokenized_false_triples, false_mask = self._process_triples(false_triples)

        # Get captions
        true_caption = tokenize(scene['true_text'], truncate=True)
        false_caption = tokenize(scene['false_text'], truncate=True)

        return {
            "image": image,
            "true_triples": tokenized_true_triples,
            "true_mask": true_mask,
            "false_triples": tokenized_false_triples,
            "false_mask": false_mask,
            "true_caption": true_caption.squeeze(0),
            "false_caption": false_caption.squeeze(0),
            "category": scene['category']
        }

class WinoGround(Dataset):
    def __init__(self, hf_dataset, transform=None, args=None):
        self.transform = transform
        self.num_triples = args.num_triples if args else 5
        self.dataset = hf_dataset
        
    def __len__(self):
        return len(self.dataset)
    
    def _process_triples(self, triples):
        triple_strings = [" ".join(t) for t in triples][:self.num_triples]
        
        num_real_triples = len(triple_strings)
        mask = torch.zeros(self.num_triples, dtype=torch.bool)
        if num_real_triples > 0:
            mask[:num_real_triples] = True
        
        padded_strings = triple_strings + [""] * (self.num_triples - num_real_triples)
        tokenized = tokenize(padded_strings, truncate=True)
        
        return tokenized, mask

    def __getitem__(self, index):
        scene = self.dataset[index]
        
        # Get image
        image_0 = scene['image_0'].convert('RGB')
        if self.transform is not None:
            image_0 = self.transform(image_0)
        image_1 = scene['image_1'].convert('RGB')
        if self.transform is not None:
            image_1 = self.transform(image_1)

        # Get triples
        triples_0 = scene['triples_0']
        triples_1 = scene['triples_1']
        
        tokenized_triples_0, mask_0 = self._process_triples(triples_0)
        tokenized_triples_1, mask_1 = self._process_triples(triples_1)

        # Get captions
        text_0 = tokenize(scene['caption_0'], truncate=True)
        text_1 = tokenize(scene['caption_1'], truncate=True)

        return {
            "image_0": image_0,
            "image_1": image_1,
            "triples_0": tokenized_triples_0,
            "mask_0": mask_0,
            "triples_1": tokenized_triples_1,
            "mask_1": mask_1,
            "text_0": text_0.squeeze(0),
            "text_1": text_1.squeeze(0),
            # "category": winoground
        }

class SugarCreppePPDataset(Dataset):
    def __init__(self, hf_dataset, transform=None, args=None):
        """
        Dataset class for SugarCrepe++ evaluation
        Args:
            hf_dataset: HuggingFace dataset (one of the 5 splits)
            transform: Image transformation function
            args: Arguments containing num_triples
        """
        self.transform = transform
        self.num_triples = args.num_triples if args else 5
        self.dataset = hf_dataset
        self.image_root = "data/coco_data/val2017/"
        
    def __len__(self):
        return len(self.dataset)
    
    def _process_triples(self, triples):
        """Process triples similar to existing implementation"""
        triple_strings = [" ".join(t) for t in triples][:self.num_triples]
        
        num_real_triples = len(triple_strings)
        mask = torch.zeros(self.num_triples, dtype=torch.bool)
        if num_real_triples > 0:
            mask[:num_real_triples] = True
        
        padded_strings = triple_strings + [""] * (self.num_triples - num_real_triples)
        tokenized = tokenize(padded_strings, truncate=True)
        
        return tokenized, mask
    
    def __getitem__(self, index):
        sample = self.dataset[index]
        
        # Load image from file path
        image_path = os.path.join(self.image_root, sample['filename'])
        image = Image.open(image_path).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        
        # Extract captions and triples
        true_caption_1 = sample['caption']
        true_caption_2 = sample['caption2']  # Second positive caption
        false_caption = sample['negative_caption']
        true_triples_1 = sample['true_triples']
        true_triples_2 = sample['true_triples2']
        false_triples = sample['false_triples']
        
        # Process triples
        tokenized_true_triples_1, true_mask_1 = self._process_triples(true_triples_1)
        tokenized_true_triples_2, true_mask_2 = self._process_triples(true_triples_2)
        tokenized_false_triples, false_mask = self._process_triples(false_triples)
        
        # Tokenize captions
        true_caption_1_tokenized = tokenize(true_caption_1, truncate=True)
        true_caption_2_tokenized = tokenize(true_caption_2, truncate=True)
        false_caption_tokenized = tokenize(false_caption, truncate=True)
        
        return {
            "image": image,
            "true_triples_1": tokenized_true_triples_1,
            "true_mask_1": true_mask_1,
            "true_triples_2": tokenized_true_triples_2,
            "true_mask_2": true_mask_2,
            "false_triples": tokenized_false_triples,
            "false_mask": false_mask,
            "true_caption_1": true_caption_1_tokenized.squeeze(0),
            "true_caption_2": true_caption_2_tokenized.squeeze(0),
            "false_caption": false_caption_tokenized.squeeze(0),
            "id": sample['id']
        }
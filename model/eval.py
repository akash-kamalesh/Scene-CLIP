import time
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from torch.nn import functional as F
from utils import image_transform, compute_logits, WinoLoss
from dataloader import Mydataset
from clip import load, tokenize
from PIL import Image
from tqdm import tqdm
import numpy as np
from tabulate import tabulate
import clip
from torch.cuda.amp import autocast
import warnings

warnings.filterwarnings('ignore')

def eval_coco_large(model, dataloader, idx2id, args):
    num_texts = len(idx2id)
    model.eval()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    start = time.time()
    all_text_features, all_image_features = [], []
    print('Evaluating COCO...')

    for i, batch in enumerate(tqdm(dataloader)):
        with torch.no_grad(), autocast():
            # Single call for true caption/triples
            image_features, caption_features, true_structural_features = model(
                batch["image"].to(device),
                batch["true_caption"].to(device),
                batch["true_triples"].to(device),
                batch["true_mask"].to(device)
            )
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            caption_features = caption_features / caption_features.norm(dim=-1, keepdim=True)
            true_structural_features = true_structural_features / true_structural_features.norm(dim=-1, keepdim=True)
            # Create fused features
            fused_text_features = caption_features + true_structural_features * 0.2
            
        all_image_features.append(image_features.cpu())
        all_text_features.append(fused_text_features.cpu())
    
    all_image_features = torch.cat(all_image_features, dim=0)
    all_text_features = torch.cat(all_text_features, dim=0)

    unique_image_ids = {}
    for i in range(num_texts):
        img_id = idx2id[i]
        if img_id not in unique_image_ids:
            unique_image_ids[img_id] = i
    
    unique_img_indices = list(unique_image_ids.values())
    unique_img_features = all_image_features[unique_img_indices]
    num_images = len(unique_img_features)

    sims_t2i = all_text_features @ unique_img_features.T

    TextRank1, TextRank5, TextRank10 = 0, 0, 0
    for i in range(num_texts):
        correct_img_id = idx2id[i]
        correct_unique_idx = list(unique_image_ids.keys()).index(correct_img_id)
        
        _, topk_indices = torch.topk(sims_t2i[i], k=10)
        
        if correct_unique_idx in topk_indices[:1]: TextRank1 += 1
        if correct_unique_idx in topk_indices[:5]: TextRank5 += 1
        if correct_unique_idx in topk_indices[:10]: TextRank10 += 1

    sims_i2t = unique_img_features @ all_text_features.T

    ImageRank1, ImageRank5, ImageRank10 = 0, 0, 0
    for i in range(num_images):
        correct_img_id = list(unique_image_ids.keys())[i]
        correct_text_indices = {idx for idx, img_id in idx2id.items() if img_id == correct_img_id}
        
        _, topk_indices_tensor = torch.topk(sims_i2t[i], k=10)
        topk_indices = topk_indices_tensor.tolist()

        if len(correct_text_indices.intersection(topk_indices[:1])) > 0: ImageRank1 += 1
        if len(correct_text_indices.intersection(topk_indices[:5])) > 0: ImageRank5 += 1
        if len(correct_text_indices.intersection(topk_indices[:10])) > 0: ImageRank10 += 1

    end = time.time()
    print("Consuming {:.2f} seconds".format(end - start))
    tr1, tr5, tr10 = TextRank1 / num_texts, TextRank5 / num_texts, TextRank10 / num_texts
    ir1, ir5, ir10 = ImageRank1 / num_images, ImageRank5 / num_images, ImageRank10 / num_images
    print(f"Text-to-Image R@1: {tr1:.4f}, R@5: {tr5:.4f}, R@10: {tr10:.4f}")
    print(f"Image-to-Text R@1: {ir1:.4f}, R@5: {ir5:.4f}, R@10: {ir10:.4f}")
    return tr1, ir1

def compute_contrastive_scores(contrastive_scores: list):
    """
    Calculates text, image, and group scores from a list of contrastive similarity dictionaries.
    """
    if not contrastive_scores:
        return {
            "text_score": 0,
            "image_score": 0,
            "group_score": 0,
            "num_samples": 0
        }

    def text_correct(result):
        return result["c0_i0"] > result["c1_i0"] and result["c1_i1"] > result["c0_i1"]

    def image_correct(result):
        return result["c0_i0"] > result["c0_i1"] and result["c1_i1"] > result["c1_i0"]

    def group_correct(result):
        return image_correct(result) and text_correct(result)

    text_correct_count = sum(1 for r in contrastive_scores if text_correct(r))
    image_correct_count = sum(1 for r in contrastive_scores if image_correct(r))
    group_correct_count = sum(1 for r in contrastive_scores if group_correct(r))

    denominator = len(contrastive_scores)
    return {
        "text_score": round(text_correct_count * 100 / denominator, 2),
        "image_score": round(image_correct_count * 100 / denominator, 2),
        "group_score": round(group_correct_count * 100 / denominator, 2),
        "num_samples": denominator,
    }

@torch.no_grad()
def eval_vismin_official_custom(model, dataloader):
    """
    Evaluates Scene-CLIP model on VisMin by strictly following the official script's logic.
    It processes each sample individually to create a 2x2 similarity matrix.
    """
    model.eval()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    start_time = time.time()
    
    categories = ['object', 'relation', 'attribute', 'counting']
    category_scores = {cat: [] for cat in categories}
    
    print('Evaluating VisMin Bench (Official Logic - Custom Model)...')
    
    for batch in tqdm(dataloader):
        batch_size = len(batch["image_0"])
        
        for i in range(batch_size):
            images = torch.stack([
                batch["image_0"][i], 
                batch["image_1"][i]
            ]).to(device)
            
            captions = torch.stack([
                batch["text_0"][i], 
                batch["text_1"][i]
            ]).to(device)
            
            triples = torch.stack([
                batch["triples_0"][i],
                batch["triples_1"][i]
            ]).to(device)
            
            masks = torch.stack([
                batch["mask_0"][i],
                batch["mask_1"][i]
            ]).to(device)

            category = batch["category"][i]

            with autocast():
                image_features, caption_features, structural_features = model(
                    images,
                    captions,
                    triples,
                    masks
                )
                
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                caption_features = caption_features / caption_features.norm(dim=-1, keepdim=True)
                structural_features = structural_features / structural_features.norm(dim=-1, keepdim=True)
                
                fused_text_features = caption_features + structural_features * 0.2
                text_features = fused_text_features / fused_text_features.norm(dim=-1, keepdim=True)

            logits_per_image = image_features @ text_features.T

            score_dict = {
                "c0_i0": logits_per_image[0, 0].item(),  # caption_0 with image_0
                "c0_i1": logits_per_image[1, 0].item(),  # caption_0 with image_1
                "c1_i0": logits_per_image[0, 1].item(),  # caption_1 with image_0
                "c1_i1": logits_per_image[1, 1].item(),  # caption_1 with image_1
            }
            
            category_scores[category].append(score_dict)

    results_table = []
    overall_scores = []
    
    for category in categories:
        scores = category_scores[category]
        if not scores:
            continue
        
        cat_results = compute_contrastive_scores(scores)
        overall_scores.extend(scores)
        
        results_table.append([
            category.capitalize(),
            f"{cat_results['text_score']:.2f}",
            f"{cat_results['image_score']:.2f}",
            f"{cat_results['group_score']:.2f}",
            cat_results['num_samples']
        ])
        
    if overall_scores:
        overall_results = compute_contrastive_scores(overall_scores)
        results_table.append([
            "Overall",
            f"{overall_results['text_score']:.2f}",
            f"{overall_results['image_score']:.2f}",
            f"{overall_results['group_score']:.2f}",
            overall_results['num_samples']
        ])

    headers = ["Category", "Text Score", "Image Score", "Group Score", "Samples"]
    print(f"\n{'='*80}")
    print("VISMIN CONTRASTIVE MATCHING RESULTS (Scene-CLIP CUSTOM MODEL)")
    print(f"{'='*80}")
    print(tabulate(results_table, headers=headers, tablefmt="fancy_grid"))
    print(f"{'='*80}\n")
    
    end_time = time.time()
    print(f"VisMin evaluation completed in {end_time - start_time:.2f} seconds")
    
    return overall_results

@torch.no_grad()
def eval_winoground_official_custom(model, dataloader):
    """
    Evaluates Scene-CLIP model on winoground by strictly following the official script's logic.
    It processes each sample individually to create a 2x2 similarity matrix.
    """
    model.eval()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    start_time = time.time()
    
    all_scores = []
    
    print('Evaluating WINOGROUND (Official Logic - Custom Model)...')
    
    for batch in tqdm(dataloader):
        batch_size = len(batch["image_0"])
        
        for i in range(batch_size):
            images = torch.stack([
                batch["image_0"][i], 
                batch["image_1"][i]
            ]).to(device)
            
            captions = torch.stack([
                batch["text_0"][i], 
                batch["text_1"][i]
            ]).to(device)
            
            triples = torch.stack([
                batch["triples_0"][i],
                batch["triples_1"][i]
            ]).to(device)
            
            masks = torch.stack([
                batch["mask_0"][i],
                batch["mask_1"][i]
            ]).to(device)

            with autocast():
                image_features, caption_features, structural_features = model(
                    images,
                    captions,
                    triples,
                    masks
                )
                
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                caption_features = caption_features / caption_features.norm(dim=-1, keepdim=True)
                structural_features = structural_features / structural_features.norm(dim=-1, keepdim=True)
                
                fused_text_features = caption_features + structural_features * 0.2
                text_features = fused_text_features / fused_text_features.norm(dim=-1, keepdim=True)

            logits_per_image = image_features @ text_features.T

            score_dict = {
                "c0_i0": logits_per_image[0, 0].item(),  # caption_0 with image_0
                "c0_i1": logits_per_image[1, 0].item(),  # caption_0 with image_1
                "c1_i0": logits_per_image[0, 1].item(),  # caption_1 with image_0
                "c1_i1": logits_per_image[1, 1].item(),  # caption_1 with image_1
            }
            
            all_scores.append(score_dict)

    overall_results = compute_contrastive_scores(all_scores)
    
    results_table = [[
        "Overall",
        f"{overall_results['text_score']:.2f}",
        f"{overall_results['image_score']:.2f}",
        f"{overall_results['group_score']:.2f}",
        overall_results['num_samples']
    ]]

    headers = ["Category", "Text Score", "Image Score", "Group Score", "Samples"]
    print(f"\n{'='*80}")
    print("WINOGROUND CONTRASTIVE MATCHING RESULTS (Scene-CLIP CUSTOM MODEL)")
    print(f"{'='*80}")
    print(tabulate(results_table, headers=headers, tablefmt="fancy_grid"))
    print(f"{'='*80}\n")
    
    end_time = time.time()
    print(f"WINOGROUND evaluation completed in {end_time - start_time:.2f} seconds")
    
    return overall_results

@torch.no_grad()
def evaluate_semantic_composition(model, loader):
    model.eval()
    
    all_img_features = []
    all_true_fused_features = []
    all_false_fused_features = []
    
    print("Extracting features...")
    for batch in tqdm(loader):
        with autocast():
            img_feat, true_caption_feat, true_structural_feat = model(
                batch["image"].cuda(),
                batch["true_caption"].cuda(),
                batch["true_triples"].cuda(),
                batch["true_mask"].cuda()
            )
            img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
            true_caption_feat = true_caption_feat / true_caption_feat.norm(dim=-1, keepdim=True)
            true_structural_feat = true_structural_feat / true_structural_feat.norm(dim=-1, keepdim=True)
            true_fused_feat = true_caption_feat + true_structural_feat * 0.2
            
            _, false_caption_feat, false_structural_feat = model(
                None,  # No image processing needed
                batch["false_caption"].cuda(),
                batch["false_triples"].cuda(),
                batch["false_mask"].cuda()
            )
            false_caption_feat = false_caption_feat / false_caption_feat.norm(dim=-1, keepdim=True)
            false_structural_feat = false_structural_feat / false_structural_feat.norm(dim=-1, keepdim=True)
            false_fused_feat = false_caption_feat + false_structural_feat * 0.2
        
        all_img_features.append(img_feat.cpu())
        all_true_fused_features.append(true_fused_feat.cpu())
        all_false_fused_features.append(false_fused_feat.cpu())
    
    all_img_features = torch.cat(all_img_features)
    all_true_fused_features = torch.cat(all_true_fused_features)
    all_false_fused_features = torch.cat(all_false_fused_features)
    
    scores_true = (all_img_features * all_true_fused_features).sum(dim=1)
    scores_false = (all_img_features * all_false_fused_features).sum(dim=1)
    
    # Calculate accuracy (how often true score > false score)
    correct_predictions = (scores_true > scores_false).sum().item()
    total_samples = len(scores_true)
    accuracy = correct_predictions / total_samples
    
    return accuracy, scores_true.numpy(), scores_false.numpy()

@torch.no_grad()
def evaluate_semantic_composition_scpp(model, loader):
    model.eval()
    
    all_img_features = []
    all_true_fused_features_1 = []
    all_true_fused_features_2 = []
    all_false_fused_features = []
    
    print("Extracting features...")
    for batch in tqdm(loader):
        with autocast():
            img_feat, true_caption_feat_1, true_structural_feat_1 = model(
                batch["image"].cuda(),
                batch["true_caption_1"].cuda(),
                batch["true_triples_1"].cuda(),
                batch["true_mask_1"].cuda()
            )
            img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
            true_caption_feat_1 = true_caption_feat_1 / true_caption_feat_1.norm(dim=-1, keepdim=True)
            true_structural_feat_1 = true_structural_feat_1 / true_structural_feat_1.norm(dim=-1, keepdim=True)
            true_fused_feat_1 = true_caption_feat_1 + true_structural_feat_1 * 0.2
            
            _, true_caption_feat_2, true_structural_feat_2 = model(
                None,  # No image processing needed
                batch["true_caption_2"].cuda(),
                batch["true_triples_2"].cuda(),
                batch["true_mask_2"].cuda()
            )
            true_caption_feat_2 = true_caption_feat_2 / true_caption_feat_2.norm(dim=-1, keepdim=True)
            true_structural_feat_2 = true_structural_feat_2 / true_structural_feat_2.norm(dim=-1, keepdim=True)
            true_fused_feat_2 = true_caption_feat_2 + true_structural_feat_2 * 0.2

            _, false_caption_feat, false_structural_feat = model(
                None,  # No image processing needed
                batch["false_caption"].cuda(),
                batch["false_triples"].cuda(),
                batch["false_mask"].cuda()
            )
            false_caption_feat = false_caption_feat / false_caption_feat.norm(dim=-1, keepdim=True)
            false_structural_feat = false_structural_feat / false_structural_feat.norm(dim=-1, keepdim=True)
            false_fused_feat = false_caption_feat + false_structural_feat * 0.2
        
        all_img_features.append(img_feat.cpu())
        all_true_fused_features_1.append(true_fused_feat_1.cpu())
        all_true_fused_features_2.append(true_fused_feat_2.cpu())
        all_false_fused_features.append(false_fused_feat.cpu())
    
    # Concatenate all features
    all_img_features = torch.cat(all_img_features)
    all_true_fused_features_1 = torch.cat(all_true_fused_features_1)
    all_true_fused_features_2 = torch.cat(all_true_fused_features_2)
    all_false_fused_features = torch.cat(all_false_fused_features)
    
    # Calculate similarity scores
    scores_true_1 = (all_img_features * all_true_fused_features_1).sum(dim=1)
    scores_true_2 = (all_img_features * all_true_fused_features_2).sum(dim=1)
    scores_false = (all_img_features * all_false_fused_features).sum(dim=1)
    
    # Calculate accuracy (how often BOTH true scores > false score)
    correct_predictions = ((scores_true_1 > scores_false) & (scores_true_2 > scores_false)).sum().item()
    total_samples = len(scores_true_1)
    accuracy = correct_predictions / total_samples
    
    return accuracy, scores_true_1.numpy(), scores_true_2.numpy(), scores_false.numpy()

@torch.no_grad()
def get_retrieval_scores_batched(model, loader, args):
    model.eval()

    all_img_features = []
    all_true_fused_features = []
    all_false_fused_features = []

    for batch in tqdm(loader):
        with autocast():
            # Get fused features for true captions/triples
            img_feat, true_caption_feat, true_structural_feat = model(
                batch["image"].cuda(),
                batch["true_caption"].cuda(),
                batch["true_triples"].cuda(),
                batch["true_mask"].cuda()
            )
            img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
            true_caption_feat = true_caption_feat / true_caption_feat.norm(dim=-1, keepdim=True)
            true_structural_feat = true_structural_feat / true_structural_feat.norm(dim=-1, keepdim=True)
            true_fused_feat = true_caption_feat + true_structural_feat * 0.2
            
            # Get fused features for false captions/triples
            _, false_caption_feat, false_structural_feat = model(
                None,  # No image processing needed
                batch["false_caption"].cuda(),
                batch["false_triples"].cuda(),
                batch["false_mask"].cuda()
            )
            false_caption_feat = false_caption_feat / false_caption_feat.norm(dim=-1, keepdim=True)
            false_structural_feat = false_structural_feat / false_structural_feat.norm(dim=-1, keepdim=True)
            false_fused_feat = false_caption_feat + false_structural_feat * 0.2
        
        all_img_features.append(img_feat.cpu())
        all_true_fused_features.append(true_fused_feat.cpu())
        all_false_fused_features.append(false_fused_feat.cpu())

    all_img_features = torch.cat(all_img_features)
    all_true_fused_features = torch.cat(all_true_fused_features)
    all_false_fused_features = torch.cat(all_false_fused_features)

    # Similarity scores
    scores_true = (all_img_features * all_true_fused_features).sum(dim=1)
    scores_false = (all_img_features * all_false_fused_features).sum(dim=1)
    
    # Stack scores for comparison: (N, 1, 2)
    scores = torch.stack([scores_false, scores_true], dim=1).unsqueeze(1)
    return scores.numpy()

drop_relations = ['adjusting',
 'attached to',
 'between',
 'bigger than',
 'biting',
 'boarding',
 'brushing',
 'chewing',
 'cleaning',
 'climbing',
 'close to',
 'coming from',
 'coming out of',
 'contain',
 'crossing',
 'dragging',
 'draped over',
 'drinking',
 'drinking from',
 'driving',
 'driving down',
 'driving on',
 'eating from',
 'eating in',
 'enclosing',
 'exiting',
 'facing',
 'filled with',
 'floating in',
 'floating on',
 'flying',
 'flying above',
 'flying in',
 'flying over',
 'flying through',
 'full of',
 'going down',
 'going into',
 'going through',
 'grazing in',
 'growing in',
 'growing on',
 'guiding',
 'hanging from',
 'hanging in',
 'hanging off',
 'hanging over',
 'higher than',
 'holding onto',
 'hugging',
 'in between',
 'jumping off',
 'jumping on',
 'jumping over',
 'kept in',
 'larger than',
 'leading',
 'leaning over',
 'leaving',
 'licking',
 'longer than',
 'looking in',
 'looking into',
 'looking out',
 'looking over',
 'looking through',
 'lying next to',
 'lying on top of',
 'making',
 'mixed with',
 'mounted on',
 'moving',
 'on the back of',
 'on the edge of',
 'on the front of',
 'on the other side of',
 'opening',
 'painted on',
 'parked at',
 'parked beside',
 'parked by',
 'parked in',
 'parked in front of',
 'parked near',
 'parked next to',
 'perched on',
 'petting',
 'piled on',
 'playing',
 'playing in',
 'playing on',
 'playing with',
 'pouring',
 'reaching for',
 'reading',
 'reflected on',
 'riding on',
 'running in',
 'running on',
 'running through',
 'seen through',
 'sitting behind',
 'sitting beside',
 'sitting by',
 'sitting in front of',
 'sitting near',
 'sitting next to',
 'sitting under',
 'skiing down',
 'skiing on',
 'sleeping in',
 'sleeping on',
 'smiling at',
 'sniffing',
 'splashing',
 'sprinkled on',
 'stacked on',
 'standing against',
 'standing around',
 'standing behind',
 'standing beside',
 'standing in front of',
 'standing near',
 'standing next to',
 'staring at',
 'stuck in',
 'surrounding',
 'swimming in',
 'swinging',
 'talking to',
 'topped with',
 'touching',
 'traveling down',
 'traveling on',
 'tying',
 'typing on',
 'underneath',
 'wading in',
 'waiting for',
 'walking across',
 'walking by',
 'walking down',
 'walking next to',
 'walking through',
 'working in',
 'working on',
 'worn on',
 'wrapped around',
 'wrapped in',    
 "by", 
 "of", 
 "near", "next to", 
 "with",
 "beside",
 "on the side of",
 "around"]

def macroacc_evaluation(scores, dataset, drop_relations=drop_relations):
    
    metrics = {"Accuracy": None}
    preds = np.argmax(np.squeeze(scores, axis=1), axis=-1)
    correct_mask = (preds == 1)
    metrics["Accuracy"] = np.mean(correct_mask)
    
    all_relations = np.array(dataset.all_relations)
    # Log the accuracy of all relations
    for relation in np.unique(all_relations):
        if relation in drop_relations:
            continue
        relation_mask = (all_relations == relation)
        if relation_mask.sum() == 0:
            continue
        metrics[f"{relation}-Acc"] = correct_mask[relation_mask].mean()
    
    return metrics

def macroacc_evaluation_attribute(scores, dataset):
    
    metrics = {"Accuracy": None}
    preds = np.argmax(np.squeeze(scores, axis=1), axis=-1)
    correct_mask = (preds == 1)
    metrics["Accuracy"] = np.mean(correct_mask)
    
    all_relations = np.array(dataset.all_attributes)
    # Log the accuracy of all relations
    for relation in np.unique(all_relations):
        relation_mask = (all_relations == relation)
        if relation_mask.sum() == 0:
            continue
        metrics[f"{relation}-Acc"] = correct_mask[relation_mask].mean()
    
    return metrics

def test_vg_relation(model, vg_relation_dataloader, vg_relation_dataset, args):
    scores = get_retrieval_scores_batched(model, vg_relation_dataloader, args)
    # np.save('/root/code/clip_order/checkpoints/case_study/relation/our_score.npy',scores)
    metrics = macroacc_evaluation(scores, vg_relation_dataset)
    all_accs = []
    for k,v in metrics.items():
        if "-Acc" in k:
            all_accs.append(v)
    acc_test_relation = np.mean(all_accs)
    print("acc_test_relation", acc_test_relation)
    return acc_test_relation


def test_vg_attribution(model, vg_attribution_dataloader, vg_attribution_dataset, args):
    scores = get_retrieval_scores_batched(model, vg_attribution_dataloader, args)
    
    metrics = macroacc_evaluation_attribute(scores, vg_attribution_dataset)
    all_accs = []
    for k,v in metrics.items():
        if "-Acc" in k:
            all_accs.append(v)
    acc_test_attribution = np.mean(all_accs)
    print("acc_test_attribution", acc_test_attribution)
    return acc_test_attribution
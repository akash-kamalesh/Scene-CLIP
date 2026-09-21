import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from datasets import load_dataset
from tqdm import tqdm
from utils import get_args, set_manualSeed, image_transform, WinoLoss, CLIPLoss, MarginLoss, TripletCLIPLoss
from dataloader import CoCoDataset_aug_update
from dataloader_downstream import VG_Relation, VG_Attribution, VisminDataset, SemanticCompositionDataset, VisMinBenchDataset,SugarCreppePPDataset
from clip import load
from model import SceneCLIP
from eval import eval_coco_large, test_vg_relation, test_vg_attribution, evaluate_semantic_composition, eval_vismin_bench, evaluate_semantic_composition_scpp
import wandb
import json
from torch.cuda.amp import GradScaler, autocast
import warnings
import os
from tabulate import tabulate

warnings.filterwarnings('ignore')

args = get_args()

wandb.init(project=args.project, name=args.name + "_lr" +str(args.lr)+ "_weight"+str(args.neg_loss_weight) + "_k" + str(args.knowledge_weight))
set_manualSeed(args)
def add_true_false_captions(batch):
    batch["true_caption"] = [caps[0] if caps else None for caps in batch["captions"]]
    batch["false_caption"] = [caps[1] if caps and len(caps) > 1 else None for caps in batch["captions"]]
    return batch

# Create checkpoint directory
checkpoint_dir = f"checkpoints/{args.name}"
os.makedirs(checkpoint_dir, exist_ok=True)

idx2id = dict()
with open("data/test_coco_aug_withneg_objectchange_relation.json", "r") as f:
    infomation = json.load(f)
for idx, item in enumerate(infomation):
    id = item['image_id']
    idx2id[idx] = id

# CLIP 
clip_model, preprocess = load("ViT-B/32", jit=False)
model = SceneCLIP(clip_model, args, agg_transformer_layers=2).cuda()

# Print trainable parameters
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total trainable parameters: {trainable_params/1e6:.2f}M")

vismin_ds = load_dataset("mair-lab/vismin",split="train")
jsonl_path = "data/vismin_triples.jsonl"  # your jsonl file
id_to_triples = {}

with open(jsonl_path, "r", encoding="utf-8") as f:
    for line in f:
        row = json.loads(line)
        image_id = row["image_id"]
        triples = row["triples"]
        id_to_triples[image_id] = triples

# ---- Step 2: Map triples into dataset ----
def add_triples(batch):
    batch["triples"] = [id_to_triples.get(img_id, []) for img_id in batch["image_id"]]
    return batch

# Assuming dataset is already loaded
vismin_ds = vismin_ds.map(
    add_triples,
    batched=True,
    num_proc=8  # adjust based on CPU cores
)
sugarcrepe = load_dataset("haideraltahan/wds_sugarcrepe",split="test")
sugarcrepe = sugarcrepe.rename_column("0.webp", "image")
sugarcrepe = sugarcrepe.rename_column("npy", "captions")
sugarcrepe = sugarcrepe.map(
    add_true_false_captions,
    batched=True,
    num_proc=4  # adjust according to CPU cores available
)
# Path to your JSONL file
jsonl_path = "data/sugarcrepe_triples.jsonl"

# Step 1 — Load all triples into memory
with open(jsonl_path, "r", encoding="utf-8") as f:
    triples_data = [json.loads(line.strip()) for line in f if line.strip()]

# Step 2 — Define map function
def add_triples(batch):
    # batch['idx'] or range(len(batch)) corresponds to the dataset indices in the batch
    batch_size = len(batch['idx']) if 'idx' in batch else len(batch[list(batch.keys())[0]])
    batch["true_triples"] = []
    batch["false_triples"] = []

    for i in range(batch_size):
        batch["true_triples"].append(triples_data[i]["true_triples"])
        batch["false_triples"].append(triples_data[i]["false_triples"])
    
    return batch

# Step 3 — Apply map
sugarcrepe = sugarcrepe.map(
    add_triples,
    batched=True,
    num_proc=4  # adjust based on CPU cores available
)
vismin_bench = load_dataset("AsphyXIA/vismin-bench", split="train")

train_dataset_vismin = VisminDataset(vismin_ds,transform=image_transform())
train_dataset_coco = VG_Attribution(data_path=args.train_path, transform=image_transform(), args=args)
train_dataset = ConcatDataset([train_dataset_vismin,train_dataset_coco])
train_dataloader = DataLoader(train_dataset, num_workers=8, batch_size=args.batch_size, shuffle=True)
test_vg_dataset = VG_Attribution(data_path=args.test_path, transform=image_transform(is_train=False), args=args)
test_vg_dataloader = DataLoader(test_vg_dataset,  num_workers=8, batch_size=args.batch_size, shuffle=False)

test_dataset = CoCoDataset_aug_update(data_path="data/test_coco_aug_withneg_objectchange_relation.json", transform=image_transform(is_train=False), args=args)
test_dataloader = DataLoader(test_dataset,  num_workers=8, batch_size=args.batch_size, shuffle=False)
vg_relation_dataset = VG_Relation(transform=image_transform(is_train=False), args=args)
vg_relation_dataloader = DataLoader(vg_relation_dataset, batch_size=args.batch_size, num_workers=8)

vismin_bench_dataset = VisMinBenchDataset(vismin_bench, transform=image_transform(is_train=False), args=args)
vismin_bench_dataloader = DataLoader(vismin_bench_dataset, batch_size=args.batch_size, num_workers=8, shuffle=False)

add_obj = sugarcrepe.filter(lambda batch: [s == "add_obj" for s in batch["split.txt"]],batched=True,num_proc=4,keep_in_memory=True)
add_att = sugarcrepe.filter(lambda batch: [s == "add_att" for s in batch["split.txt"]],batched=True,num_proc=4,keep_in_memory=True)
replace_obj = sugarcrepe.filter(lambda batch: [s == "replace_obj" for s in batch["split.txt"]],batched=True,num_proc=4,keep_in_memory=True)
replace_att = sugarcrepe.filter(lambda batch: [s == "replace_att" for s in batch["split.txt"]],batched=True,num_proc=4,keep_in_memory=True)
replace_rel = sugarcrepe.filter(lambda batch: [s == "replace_rel" for s in batch["split.txt"]],batched=True,num_proc=4,   keep_in_memory=True)
swap_obj = sugarcrepe.filter(lambda batch: [s == "swap_obj" for s in batch["split.txt"]],batched=True,num_proc=4,   keep_in_memory=True)
swap_att = sugarcrepe.filter(lambda batch: [s == "swap_att" for s in batch["split.txt"]],batched=True,num_proc=4,   keep_in_memory=True)

add_obj = SemanticCompositionDataset(hf_dataset=add_obj,transform=image_transform())
add_att = SemanticCompositionDataset(hf_dataset=add_att,transform=image_transform())
replace_obj = SemanticCompositionDataset(hf_dataset=replace_obj,transform=image_transform())
replace_att = SemanticCompositionDataset(hf_dataset=replace_att,transform=image_transform())
replace_rel = SemanticCompositionDataset(hf_dataset=replace_rel,transform=image_transform())
swap_obj = SemanticCompositionDataset(hf_dataset=swap_obj,transform=image_transform())
swap_att = SemanticCompositionDataset(hf_dataset=swap_att,transform=image_transform())

add_obj_dl = DataLoader(add_obj,batch_size=32,shuffle=False,num_workers=4)
add_att_dl = DataLoader(add_att,batch_size=32,shuffle=False,num_workers=4)
replace_obj_dl = DataLoader(replace_obj,batch_size=32,shuffle=False,num_workers=4)
replace_att_dl = DataLoader(replace_att,batch_size=32,shuffle=False,num_workers=4)
replace_rel_dl = DataLoader(replace_rel,batch_size=32,shuffle=False,num_workers=4)
swap_obj_dl = DataLoader(swap_obj,batch_size=32,shuffle=False,num_workers=4)
swap_att_dl = DataLoader(swap_att,batch_size=32,shuffle=False,num_workers=4)

# Load SugarCrepe++ datasets
sugarcrepe_pp_rep_att = load_dataset("AsphyXIA/sugarcrepe_pp", split='rep_att')
sugarcrepe_pp_rep_obj = load_dataset("AsphyXIA/sugarcrepe_pp", split='rep_obj')
sugarcrepe_pp_rep_rel = load_dataset("AsphyXIA/sugarcrepe_pp", split='rep_rel')
sugarcrepe_pp_swap_att = load_dataset("AsphyXIA/sugarcrepe_pp", split='swap_att')
sugarcrepe_pp_swap_obj = load_dataset("AsphyXIA/sugarcrepe_pp", split='swap_obj')

# Create dataset instances
sugarcrepe_pp_rep_att_ds = SugarCreppePPDataset(hf_dataset=sugarcrepe_pp_rep_att, transform=image_transform())
sugarcrepe_pp_rep_obj_ds = SugarCreppePPDataset(hf_dataset=sugarcrepe_pp_rep_obj, transform=image_transform())
sugarcrepe_pp_rep_rel_ds = SugarCreppePPDataset(hf_dataset=sugarcrepe_pp_rep_rel, transform=image_transform())
sugarcrepe_pp_swap_att_ds = SugarCreppePPDataset(hf_dataset=sugarcrepe_pp_swap_att, transform=image_transform())
sugarcrepe_pp_swap_obj_ds = SugarCreppePPDataset(hf_dataset=sugarcrepe_pp_swap_obj, transform=image_transform())

# Create dataloaders
sugarcrepe_pp_rep_att_dl = DataLoader(sugarcrepe_pp_rep_att_ds, batch_size=32, shuffle=False, num_workers=4)
sugarcrepe_pp_rep_obj_dl = DataLoader(sugarcrepe_pp_rep_obj_ds, batch_size=32, shuffle=False, num_workers=4)
sugarcrepe_pp_rep_rel_dl = DataLoader(sugarcrepe_pp_rep_rel_ds, batch_size=32, shuffle=False, num_workers=4)
sugarcrepe_pp_swap_att_dl = DataLoader(sugarcrepe_pp_swap_att_ds, batch_size=32, shuffle=False, num_workers=4)
sugarcrepe_pp_swap_obj_dl = DataLoader(sugarcrepe_pp_swap_obj_ds, batch_size=32, shuffle=False, num_workers=4)

loss_margin = MarginLoss(margin=0.1) # For true vs false
loss_itc = WinoLoss(margin=0.1)
scaler = GradScaler()

optimizer_params = [
    {'params': model.clip.parameters()},
]
if not args.mean_pool:
    optimizer_params.append({'params': model.triple_aggregator.parameters()})

optimizer = torch.optim.AdamW(optimizer_params, lr=args.lr, weight_decay=args.weight_decay)
best_avg_vg_score = 0.0

for epoch in range(args.epoch):
    model.train()
    for i, batch in enumerate(tqdm(train_dataloader, total=len(train_dataloader))):
        if (i+1) % 2099 == 0:
            model.eval()
            # eval task1
            t1, i1 = eval_coco_large(model, test_dataloader, idx2id, args)
            wandb.log({"COCOTextRank1": t1})
            wandb.log({"COCOImageRank1": i1})

            # eval task2
            acc_test_attribution = test_vg_attribution(model, test_vg_dataloader, test_vg_dataset, args)
            wandb.log({"acc_test_attribution": acc_test_attribution})           

            acc_test_relation = test_vg_relation(model, vg_relation_dataloader, vg_relation_dataset, args)
            wandb.log({"acc_test_relation": acc_test_relation})

            # Evaluate semantic composition tasks
            add_obj_acc,_,_ = evaluate_semantic_composition(model,add_obj_dl)
            add_att_acc,_,_ = evaluate_semantic_composition(model,add_att_dl)
            replace_obj_acc,_,_ = evaluate_semantic_composition(model,replace_obj_dl)
            replace_att_acc,_,_ = evaluate_semantic_composition(model,replace_att_dl)
            replace_rel_acc,_,_ = evaluate_semantic_composition(model,replace_rel_dl)
            swap_obj_acc,_,_ = evaluate_semantic_composition(model,swap_obj_dl)
            swap_att_acc,_,_ = evaluate_semantic_composition(model,swap_att_dl)

            # Pretty print semantic composition results using tabulate
            semantic_results = [
                ["add_obj", f"{add_obj_acc:.4f}"],
                ["add_att", f"{add_att_acc:.4f}"],
                ["replace_obj", f"{replace_obj_acc:.4f}"],
                ["replace_att", f"{replace_att_acc:.4f}"],
                ["replace_rel", f"{replace_rel_acc:.4f}"],
                ["swap_obj", f"{swap_obj_acc:.4f}"],
                ["swap_att", f"{swap_att_acc:.4f}"],
                ["", ""],  # Empty row for separation
                ["AVERAGE", f"{(add_obj_acc + add_att_acc + replace_obj_acc + replace_att_acc + replace_rel_acc + swap_obj_acc + swap_att_acc) / 7:.4f}"]
            ]
            
            print(f"\n{'='*60}")
            print(f"SEMANTIC COMPOSITION RESULTS - Epoch {epoch}, Step {i+1}")
            print(f"{'='*60}")
            print(tabulate(semantic_results, 
                        headers=["SugarCrepe Type", "Accuracy"], 
                        tablefmt="fancy_grid",
                        floatfmt=".4f"))
            print(f"{'='*60}\n")

            # Evaluate on SugarCrepe++ datasets
            print("\n" + "="*50)
            print("EVALUATING ON SUGARCREPE++ DATASETS")
            print("="*50)

            sugarcrepe_pp_results = {}

            # Evaluate each SugarCrepe++ split
            sugarcrepe_pp_splits = [
                ("rep_att", sugarcrepe_pp_rep_att_dl),
                ("rep_obj", sugarcrepe_pp_rep_obj_dl), 
                ("rep_rel", sugarcrepe_pp_rep_rel_dl),
                ("swap_att", sugarcrepe_pp_swap_att_dl),
                ("swap_obj", sugarcrepe_pp_swap_obj_dl)
            ]

            for split_name, dataloader in sugarcrepe_pp_splits:
                print(f"\nEvaluating SugarCrepe++ {split_name}...")
                accuracy, _, _, _ = evaluate_semantic_composition_scpp(model, dataloader)
                sugarcrepe_pp_results[split_name] = accuracy
                print(f"SugarCrepe++ {split_name} Accuracy: {accuracy:.4f}")

            # Pretty print all SugarCrepe++ results
            print("\n" + "="*50)
            print("SUGARCREPE++ RESULTS SUMMARY")
            print("="*50)

            sugarcrepe_pp_table = [
                ["Split", "Accuracy"],
                ["rep_att", f"{sugarcrepe_pp_results['rep_att']:.4f}"],
                ["rep_obj", f"{sugarcrepe_pp_results['rep_obj']:.4f}"],
                ["rep_rel", f"{sugarcrepe_pp_results['rep_rel']:.4f}"],
                ["swap_att", f"{sugarcrepe_pp_results['swap_att']:.4f}"],
                ["swap_obj", f"{sugarcrepe_pp_results['swap_obj']:.4f}"],
                ["Average", f"{sum(sugarcrepe_pp_results.values()) / len(sugarcrepe_pp_results):.4f}"]
            ]

            print(tabulate(sugarcrepe_pp_table, headers="firstrow", tablefmt="grid"))

            vismin_results = eval_vismin_bench(model, vismin_bench_dataloader, args)
            vg_avg = (acc_test_attribution + acc_test_relation) / 2
            sugarcrepe_avg = (add_obj_acc + add_att_acc + replace_obj_acc + replace_att_acc + replace_rel_acc + swap_obj_acc + swap_att_acc) / 7
            sugarcrepe_pp_avg = sum(sugarcrepe_pp_results.values()) / len(sugarcrepe_pp_results)
            overall_avg_score = (vg_avg + sugarcrepe_avg + sugarcrepe_pp_avg) / 3
            wandb_log_dict = {
                "sc_add_obj_acc": add_obj_acc,
                "sc_add_att_acc": add_att_acc,
                "sc_replace_obj_acc": replace_obj_acc,
                "sc_replace_att_acc": replace_att_acc,
                "sc_replace_rel_acc": replace_rel_acc,
                "sc_swap_obj_acc": swap_obj_acc,
                "sc_swap_att_acc": swap_att_acc,
                "vg_avg": vg_avg,
                "sugarcrepe_avg": sugarcrepe_avg,
                "scpp_rep_att_acc": sugarcrepe_pp_results['rep_att'],
                "scpp_rep_obj_acc": sugarcrepe_pp_results['rep_obj'],
                "scpp_rep_rel_acc": sugarcrepe_pp_results['rep_rel'],
                "scpp_swap_att_acc": sugarcrepe_pp_results['swap_att'],
                "scpp_swap_obj_acc": sugarcrepe_pp_results['swap_obj'],
                "sugarcrepe_pp_avg": sugarcrepe_pp_avg,
                "overall_avg_score": overall_avg_score,
                "vismin_t2i_r1": vismin_results['overall_t2i_r1'],
                "vismin_i2t_r1": vismin_results['overall_i2t_r1']
            }

            # Add category-specific results to wandb logging
            for category, results in vismin_results['category_results'].items():
                wandb_log_dict[f"vismin_{category}_t2i_r1"] = results['t2i_r1']
                wandb_log_dict[f"vismin_{category}_i2t_r1"] = results['i2t_r1']

            wandb.log(wandb_log_dict)

            if overall_avg_score > best_avg_vg_score:
                best_avg_vg_score = overall_avg_score
                best_model_path = os.path.join(checkpoint_dir, "model_best.pt")
                print(f"\nNew best overall score: {best_avg_vg_score:.4f}!")
                print(f"  - VG Average: {vg_avg:.4f}")
                print(f"  - SugarCrepe Average: {sugarcrepe_avg:.4f}")
                print(f"  - SugarCrepe++ Average: {sugarcrepe_pp_avg:.4f}")
                print(f"Saving model to {best_model_path}\n")
                torch.save(model.state_dict(), best_model_path)
            
            model.train()
        
        optimizer.zero_grad()
        
        img = batch["image"].cuda()
        true_caption = batch["true_caption"].cuda()
        false_caption = batch["false_caption"].cuda()
        true_triples = batch["true_triples"].cuda()
        true_mask = batch["true_mask"].cuda()
        false_triples = batch["false_triples"].cuda()
        false_mask = batch["false_mask"].cuda()
                
        with autocast():
            # First call: get image, true caption, and true structural features
            image_features, true_caption_features, true_structural_features = model(
                img, true_caption, true_triples, true_mask
            )
            
            # Second call: get false caption and false structural features (no image processing)
            _, false_caption_features, false_structural_features = model(
                None, false_caption, false_triples, false_mask  # image=None to save computation
            )
            
            # Normalize for loss calculation
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            true_structural_features = true_structural_features / true_structural_features.norm(dim=-1, keepdim=True)
            true_caption_features = true_caption_features / true_caption_features.norm(dim=-1, keepdim=True)
            false_caption_features = false_caption_features / false_caption_features.norm(dim=-1, keepdim=True)
            false_structural_features = false_structural_features / false_structural_features.norm(dim=-1, keepdim=True)
            
            # Create fused features for true pairs only
            fused_true_features = true_caption_features + true_structural_features * 0.2
            
            # Loss Calculation
            # 1. InfoNCE Loss on fused features (image vs fused text)
            if epoch == 0 and i < 100:
                itc_loss = loss_itc(image_features, fused_true_features, 1, False)
            else:
                itc_loss = loss_itc(image_features, fused_true_features, 1, True)
            
            # 2. Margin Loss (image vs true caption vs false caption)
            margin_loss = loss_margin(image_features, true_caption_features, false_caption_features, 1) * args.neg_loss_weight

            total_loss = itc_loss + margin_loss
        
        scaler.scale(total_loss).backward()
        scaler.step(optimizer)
        scaler.update()

        if i % 10 == 0:
            print('Epoch:{},  step:{},  loss:{:.4f}, itc_loss:{:.4f}, margin_loss:{:.4f}'.format(
                epoch, i, total_loss.item(), itc_loss.item(), margin_loss.item()))
            wandb.log({
                "Loss": total_loss.item(), 
                "ITCLoss": itc_loss.item(), 
                "MarginLoss": margin_loss.item()
            })

    # Save checkpoint at the end of every epoch
    epoch_save_path = os.path.join(checkpoint_dir, f"model_epoch_{epoch}.pt")
    print(f"\nEpoch {epoch} finished. Saving checkpoint to {epoch_save_path}\n")
    torch.save(model.state_dict(), epoch_save_path)

    print('----------------------this is{}_th epoch----------------------------'.format(epoch))



        
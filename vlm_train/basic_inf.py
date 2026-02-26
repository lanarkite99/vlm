import torch
import torch.nn.functional as F
import os,random

from datasets.coco_subset_dataset import CocoSubsetCaptionDataset
from nn_arch.qformer import QFormer
from utils.calculate_clip_loss import calc_clip_loss
from utils.calc_recall import calc_recall
from utils.utils import*

def main():
    device=torch.device("cuda" if torch.cuda.is_available()else "cpu")
    print(f"Device: {device}")
    model_path="models/trained_qformer_50k_unimodal_fresh/best"
    output_path="outputs/basic_inf"
    num_samples=10
    recall_max_samples=500

    #load dataset vit and tokenizer
    print('loading dataset...')
    dataset=CocoSubsetCaptionDataset(
    tokenizer='distilbert-base-uncased',
    )
    print(f"loading qformer from {model_path}...")
    if not os.path.exists(model_path):
        print(f"model not found at {model_path}")
        return
    qformer=QFormer.from_pretrained(model_path)
    qformer.to(device)
    qformer.eval()

    #5x5 grid of images
    dataset_len=len(dataset)
    indices=random.sample(range(dataset_len),num_samples)
    print('selected indices:',indices)

    samples=[]
    for idx in indices:
        img_tensor,txt=dataset[idx]
        img_path=dataset._example[idx].image_path
        with Image.open(img_path)as img:
            og_img=img.convert('RGB')
        samples.append({'img_tensor':img_tensor,'og_img':og_img,'txt':txt,'idx':idx})

        #NxN similarity mat.
    print('computing similarity matrix...')
    scores_matrix=torch.zeros((num_samples,num_samples))

    for row_idx,test_sample in enumerate(samples):
        text_inputs=dataset.tokenizer(test_sample['txt'],return_tensors='pt',padding=True,truncation=True,max_length=512)
        ip_ids=text_inputs['input_ids'].to(device)
        attention_mask=text_inputs['attention_mask'].to(device)

        for col_idx,img_sample in enumerate(samples):
            img_feats=img_sample['image_tensor'].to(device)
            with torch.no_grad():
                q_out,t_out=qformer(img_feats,ip_ids,attention_mask,'uni_modal')
            q_norm=F.normalize(q_out,dim=-1)
            t_norm=F.normalize(t_out,dim=-1)
            similarity=(q_norm@t_norm.t()).item()
            scores_matrix[row_idx,col_idx]=similarity
    print(f'calc recall@k... {recall_max_samples} samples')

    #test loader
    _,test_loader=get_dataloader(batch_size=16)
    recall_metrics=calc_recall(model=qformer,dataloader=test_loader,
    device=device,k_values=[1,5,10],max_samples=recall_max_samples)
    print('recall metrics:',recall_metrics)

    print('creating grid of images...')
    create_grid_of_images(samples,scores_matrix,recall_metrics,output_path)
    print(f'grid saved to {output_path}')

if __name__=='__main__':
    main()

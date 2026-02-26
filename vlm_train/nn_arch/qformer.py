import torch
import torch.nn as nn
import copy,os,json
from transformers import DistilBertConfig,DistilBertModel
from typing import Literal
import numpy as np

def create_attention_mask(
B,
I,
text_presence_mask,
mode:Literal["uni_modal","multi_modal","multi_modal_causal"]="uni_modal",
):

    T=text_presence_mask.size(1)
    device=text_presence_mask.device

    mask=torch.zeros(
    B,T+I,T+I,device=device,dtype=torch.bool
    )

    img_self=torch.ones(B,I,I,device=device,dtype=torch.bool)
    text_self=torch.ones(B,T,T,device=device,dtype=torch.bool)

    if mode=="multi_modal_causal":
        text_self=torch.tril(text_self)

    if mode=="uni_modal":
        multimodal_cross_fn=torch.zeros
    else:
        multimodal_cross_fn=torch.ones

    img_cross=multimodal_cross_fn(B,T,I,device=device,dtype=torch.bool)
    text_cross=multimodal_cross_fn(B,I,T,device=device,dtype=torch.bool)

    mask[:,:T,:T]=text_self
    mask[:,-I:,-I:]=img_self
    mask[:,:T,-I:]=img_cross
    mask[:,-I:,:T]=text_cross

    presence_mask=torch.cat(
    [text_presence_mask,torch.ones(B,I,dtype=torch.bool,device=device)],dim=1
    )
    presence_mask=presence_mask.unsqueeze(2)&presence_mask.unsqueeze(1)
    mask=mask&presence_mask
    return mask.unsqueeze(1)


def create_distilbert_attention_mask(B,I,text_presence_mask):
    query_mask=torch.ones(B,I,dtype=torch.bool,device=text_presence_mask.device)
    return torch.cat([text_presence_mask.to(torch.bool),query_mask],dim=1)

class CrossAttentionBlock(nn.Module):
    def __init__(self,hidden_size,n_heads):
        super().__init__()
        self.cross_attn=nn.MultiheadAttention(
        hidden_size,n_heads,batch_first=True
        )
        self.layernorm=nn.LayerNorm(hidden_size)
        self.ffn=nn.Sequential(
        nn.Linear(hidden_size,hidden_size*4),
        nn.GELU(),
        nn.Linear(hidden_size*4,hidden_size),
        )
        self.ln2=nn.LayerNorm(hidden_size)
    def forward(self,x_queries,kv):
        attn_out,_=self.cross_attn(x_queries,kv,kv)
        x=self.layernorm(x_queries+attn_out)
        ffn_out=self.ffn(x)
        x=self.ln2(x+ffn_out)
        return x

class QFormer(nn.Module):
    def __init__(self,bert_model:DistilBertModel,n_queries=64,cross_every=2,n_heads=12):
        super().__init__()
        self.cross_every=cross_every
        self.n_heads=n_heads
        self.n_queries=n_queries

        self.bert_config=bert_model.config
        cfg:DistilBertConfig=bert_model.config
        self.hidden_size=cfg.hidden_size
        self.embeddings=copy.deepcopy(bert_model.embeddings)
        self.encoder_layers=nn.ModuleList([copy.deepcopy(layer)for layer in bert_model.transformer.layer])
        self.cross_blocks=nn.ModuleDict()
        for i in range(len(self.encoder_layers)):
            if(i%self.cross_every)==(cross_every-1):
                self.cross_blocks[str(i)]=CrossAttentionBlock(self.hidden_size,n_heads)
        self.query_embeddings=nn.Parameter(torch.randn(1,n_queries,self.hidden_size))

    def save_pretrained(self,save_dir:str):
        os.makedirs(save_dir,exist_ok=True)
        config={"n_queries":self.n_queries,
        "cross_every":self.cross_every,
        "n_heads":self.n_heads,
        "bert_config":self.bert_config.to_dict(),}
        with open(os.path.join(save_dir,"config.json"),"w")as f:
            json.dump(config,f,indent=2)
        torch.save(self.state_dict(),os.path.join(save_dir,"pytorch_model.bin"))

    @classmethod
    def from_pretrained(cls,load_dir):
        with open(os.path.join(load_dir,"config.json"),"r")as f:
            config=json.load(f)
        bert_config_dict=config.pop("bert_config")
        bert_config=DistilBertConfig.from_dict(bert_config_dict)

        bert_model=DistilBertModel(bert_config)
        model=cls(bert_model=bert_model,**config)
        state_dict_path=os.path.join(load_dir,"pytorch_model.bin")
        state_dict=torch.load(state_dict_path,map_location="cpu")
        model.load_state_dict(state_dict)
        return model

    def get_grouped_parameters(self):
        params={"default":[],"cross_blocks":[],"query_embeddings":[]}
        for name,param in self.named_parameters():
            if "query_embeddings" in name:
                params["query_embeddings"].append(param)
            elif "cross_blocks" in name:
                params["cross_blocks"].append(param)
            else:
                params["default"].append(param)
        return params

    def encode_image(self,image_features):
        B=image_features.shape[0]
        x=self.query_embeddings.expand(B,-1,-1)
        for i,layer in enumerate(self.encoder_layers):
            x=layer(x)
            if str(i)in self.cross_blocks:
                x=self.cross_blocks[str(i)](x,image_features)
        return x,x.mean(dim=1)

    def forward(
    self,
    image_features,
    text_input_ids,
    text_attention_mask=None,
    attention_mode:Literal[
    "uni_modal","multi_modal","multi_modal_causal"
    ]="uni_modal",
    ):

        if text_attention_mask is None:
            text_attention_mask=torch.ones_like(text_input_ids)

        text_attention_mask=text_attention_mask.to(torch.bool)

        if image_features.dim()!=3:
            raise ValueError(
            f"image_features must be rank-3 [B, N, D], got shape {tuple(image_features.shape)}"
            )

        txt_emb=self.embeddings(input_ids=text_input_ids)
        B=txt_emb.size(0)

        if image_features.size(0)!=B:
            raise ValueError(
            f"Batch size mismatch: image_features has B={image_features.size(0)}, "
            f"text_input_ids has B={B}"
            )
        if image_features.size(-1)!=self.hidden_size:
            raise ValueError(
            f"Feature dim mismatch: image_features dim={image_features.size(-1)} "
            f"but qformer hidden_size={self.hidden_size}"
            )

        queries=self.query_embeddings.expand(B,-1,-1)

        x=torch.cat([txt_emb,queries],dim=1)

        attention_mask=create_attention_mask(
        B=B,
        I=self.n_queries,
        text_presence_mask=text_attention_mask,
        mode=attention_mode,
        )

        for i,layer in enumerate(self.encoder_layers):

            x=layer(x,attention_mask=attention_mask)

            if str(i)in self.cross_blocks:
                queries=x[:,-self.n_queries:]
                txt_emb=x[:,:-self.n_queries]
                queries=self.cross_blocks[str(i)](queries,image_features)
                x=torch.cat([txt_emb,queries],dim=1)

        queries=x[:,-self.n_queries:]
        txt_emb=x[:,:-self.n_queries]

        return queries.mean(dim=1),txt_emb[:,0]

if __name__=="__main__":
    bert=DistilBertModel.from_pretrained("distilbert-base-uncased")
    batch_size=2
    seq_len=5
    num_patches=49
    d_image=768
    n_queries=3
    input_ids=torch.randint(0,30522,(batch_size,seq_len),dtype=torch.long)

    from transformers import ViTImageProcessor,ViTModel
    from PIL import Image

    vit_name="google/vit-base-patch16-224"
    processor=ViTImageProcessor.from_pretrained(vit_name)
    vit=ViTModel.from_pretrained(vit_name)

    dummy_images=[
    Image.fromarray(np.uint8(np.random.rand(224,224,3)*255))
    for _ in range(batch_size)
    ]
    inputs=processor(images=dummy_images,return_tensors="pt")

    with torch.no_grad():
        vit_out=vit(**inputs).last_hidden_state

    image_feats=vit_out


    text_attention_mask=torch.tensor(
    [[1,1,1,0,0],[1,1,1,1,0]],dtype=torch.bool
    )
    qformer=QFormer(bert,n_queries=n_queries)

    qformer.eval()
    with torch.no_grad():
        out_queries,out_text=qformer(
        image_feats,
        input_ids,
        text_attention_mask,
        attention_mode="multi_modal_causal",
        )

    print("Original Output shapes:",out_queries.shape,out_text.shape)

    save_dir="tmp_qformer_checkpoints"
    print(f"Saving model to {save_dir}...")
    qformer.save_pretrained(save_dir)

    print(f"Loading model from {save_dir}...")
    loaded_qformer=QFormer.from_pretrained(save_dir)

    sum_orig=sum(p.sum().item()for p in qformer.parameters())
    sum_loaded=sum(p.sum().item()for p in loaded_qformer.parameters())
    print(f"Sum of original weights: {sum_orig}")
    print(f"Sum of loaded weights: {sum_loaded}")
    print(f"Weight sum difference: {abs(sum_orig - sum_loaded)}")

    loaded_qformer.eval()
    with torch.no_grad():
        out_queries_loaded,out_text_loaded=loaded_qformer(
        image_feats,
        input_ids,
        text_attention_mask,
        attention_mode="multi_modal_causal",
        )

    print("Loaded Output shapes:",out_queries_loaded.shape,out_text_loaded.shape)

    diff_queries=(out_queries-out_queries_loaded).abs().max()
    diff_text=(out_text-out_text_loaded).abs().max()

    print(f"Max difference in queries: {diff_queries}")
    print(f"Max difference in text: {diff_text}")

    if diff_queries<1e-6 and diff_text<1e-6:
        print("SUCCESS: Model saved and loaded correctly!")
    else:
        print("FAILURE: Model outputs mismatch after loading.")

        # Cleanup
    import shutil

    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
        # bert = DistilBertModel.from_pretrained("distilbert-base-uncased")
        # qformer = QFormer(bert, n_queries=64)

        # image_feats = torch.randn(2, 197, 768)
        # input_ids = torch.randint(0, 30522, (2, 5))
        # mask = torch.ones_like(input_ids)

        # qformer.eval()
        # with torch.no_grad():
        #     out_q, out_t = qformer(
        #         image_feats,
        #         input_ids,
        #         mask,
        #         "multi_modal",
        #     )

        # print("Query pooled shape:", out_q.shape)
        # print("Text pooled shape:", out_t.shape)




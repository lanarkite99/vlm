from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from peft import LoraConfig,TaskType,get_peft_model
from transformers import AutoModelForCausalLM,AutoTokenizer,ViTModel

from vlm_train.nn_arch.qformer import QFormer


class LM_2_VLM(nn.Module):
    def __init__(
    self,
    model_name:str,
    qformer_model_path:str,
    pad_token_id:int,
    vit_name:str="google/vit-base-patch16-224",
    max_text_len:int=128,
    lora_r:int=64,
    lora_alpha:int=128,
    lora_dropout:float=0.05,
    lora_target_modules:Optional[list[str]]=None,
    train_llm:bool=True,
    ):
        super().__init__()
        self.max_text_len=max_text_len

        self.vit=ViTModel.from_pretrained(vit_name)
        self.vit.eval()
        for p in self.vit.parameters():
            p.requires_grad=False

        self.qformer=QFormer.from_pretrained(qformer_model_path)

        base_llm=AutoModelForCausalLM.from_pretrained(model_name)
        self.tokenizer=AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token=self.tokenizer.eos_token
        base_llm.config.pad_token_id=pad_token_id

        if lora_target_modules is None:
            lora_target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "up_proj",
            "down_proj",
            "gate_proj",
            ]

        peft_cfg=LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=lora_target_modules,
        )
        self.llm=get_peft_model(base_llm,peft_cfg)
        if not train_llm:
            for p in self.llm.parameters():
                p.requires_grad=False

        self.adapter=nn.Linear(self.qformer.hidden_size,self.llm.config.hidden_size)

    def get_grouped_params(self):
        qparams=self.qformer.get_grouped_parameters()
        groups={
        "qformer_default":qparams["default"],
        "qformer_cross":qparams["cross_blocks"],
        "qformer_query":qparams["query_embeddings"],
        "adapter":list(self.adapter.parameters()),
        "llm_trainable":[p for p in self.llm.parameters()if p.requires_grad],
        }
        return groups

    def _build_lm_inputs(self,prefix_texts,assistant_texts,vision_embeds,device):
        emb_layer=self.llm.get_input_embeddings()
        eos_id=self.tokenizer.eos_token_id

        batch_embeds=[]
        batch_labels=[]
        batch_attn=[]
        max_len=0

        for i,(prefix,answer)in enumerate(zip(prefix_texts,assistant_texts)):
        # Small chat-style prompt format.
            prompt=f"<|user|>\n{prefix}\n<|assistant|>\n"
            p_ids=self.tokenizer(prompt,add_special_tokens=False)["input_ids"]
            a_ids=self.tokenizer(answer,add_special_tokens=False)["input_ids"]
            if eos_id is not None:
                a_ids=a_ids+[eos_id]

                # Bound sequence length for predictable memory.
            p_ids=p_ids[:self.max_text_len//2]
            a_ids=a_ids[:self.max_text_len-len(p_ids)]

            text_ids=p_ids+a_ids
            text_ids_t=torch.tensor(text_ids,dtype=torch.long,device=device)
            text_emb=emb_layer(text_ids_t)

            vis=vision_embeds[i]
            full_emb=torch.cat([vis,text_emb],dim=0)

            labels=[-100]*vis.size(0)+[-100]*len(p_ids)+a_ids
            labels_t=torch.tensor(labels,dtype=torch.long,device=device)
            attn_t=torch.ones(full_emb.size(0),dtype=torch.long,device=device)

            batch_embeds.append(full_emb)
            batch_labels.append(labels_t)
            batch_attn.append(attn_t)
            max_len=max(max_len,full_emb.size(0))

        hidden=batch_embeds[0].size(-1)
        bsz=len(batch_embeds)
        pad_emb=torch.zeros(
        bsz,max_len,hidden,dtype=batch_embeds[0].dtype,device=device
        )
        pad_labels=torch.full((bsz,max_len),-100,dtype=torch.long,device=device)
        pad_attn=torch.zeros((bsz,max_len),dtype=torch.long,device=device)

        for i in range(bsz):
            n=batch_embeds[i].size(0)
            pad_emb[i,:n]=batch_embeds[i]
            pad_labels[i,:n]=batch_labels[i]
            pad_attn[i,:n]=batch_attn[i]

        return pad_emb,pad_attn,pad_labels

    def forward(self,pixel_values,prefix,assistant_prompt):
        device=pixel_values.device
        with torch.no_grad():
            image_features=self.vit(pixel_values=pixel_values).last_hidden_state

        query_tokens,_=self.qformer.encode_image(image_features)
        vision_embeds=self.adapter(query_tokens)

        inputs_embeds,attention_mask,labels=self._build_lm_inputs(
        prefix_texts=prefix,
        assistant_texts=assistant_prompt,
        vision_embeds=vision_embeds,
        device=device,
        )
        out=self.llm(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        labels=labels,
        )
        return out

    def save_checkpoint(self,save_dir:str):
        save_path=Path(save_dir)
        save_path.mkdir(parents=True,exist_ok=True)
        self.qformer.save_pretrained(str(save_path/"qformer"))
        torch.save(self.adapter.state_dict(),save_path/"adapter.pt")
        torch.save(self.llm.state_dict(),save_path/"llm_state.pt")
        # PEFT save_pretrained stores LoRA adapter weights/config.
        self.llm.save_pretrained(str(save_path/"llm"))
        self.tokenizer.save_pretrained(str(save_path/"llm"))

    def load_checkpoint(self,load_dir:str,map_location:Optional[str]=None):
        load_path=Path(load_dir)
        qformer_dir=load_path/"qformer"
        if qformer_dir.exists():
            loaded_qformer=QFormer.from_pretrained(str(qformer_dir))
            self.qformer.load_state_dict(loaded_qformer.state_dict())
        if(load_path/"adapter.pt").exists():
            self.adapter.load_state_dict(
            torch.load(load_path/"adapter.pt",map_location=map_location)
            )
        if(load_path/"llm_state.pt").exists():
            self.llm.load_state_dict(
            torch.load(load_path/"llm_state.pt",map_location=map_location),
            strict=False,
            )

from manim import *


BG="#08090c"
TXT="#e7e7e7"
STROKE="#b8b8b8"
BLOCK="#2f3138"
QF="#245f74"
LORA="#2f7a93"
CONCAT="#c8c8c8"
TOK_BLUE="#63bfe2"
TOK_RED="#e78879"
TOK_GRAY="#8f8f8f"
FONT="DejaVu Sans"


def make_block(label,w,h,fill=BLOCK,fs=18,color=TXT):
    rect=Rectangle(width=w,height=h,stroke_color=STROKE,stroke_width=1.3,fill_color=fill,fill_opacity=1.0)
    text=Text(label,font=FONT,font_size=fs,color=color).move_to(rect.get_center())
    return VGroup(rect,text)


def make_stack(n,color,w=0.33,h=0.10,g=0.045):
    s=VGroup()
    for i in range(n):
        c=Rectangle(width=w,height=h,stroke_color=STROKE,stroke_width=0.6,fill_color=color,fill_opacity=0.96)
        c.move_to([0,-i*(h+g),0])
        s.add(c)
    return s


class VLMArchitecture3B1B(Scene):
    def construct(self):
        self.camera.background_color=BG
        # Force a wide composition so it does not compress in the center.
        self.camera.frame_width=36
        self.camera.frame_height=14

        title=Text("Two-Stage VLM Pipeline",font=FONT,font_size=34,color=TXT).move_to([0,5.8,0])
        prompt=Text('"Describe this image"',font=FONT,font_size=30,color=TXT).move_to([0,4.7,0])
        self.add(title,prompt)

        # Manual x positions for wide spacing.
        y0=-0.7
        x={
            "inp":-15.3,
            "vit":-12.1,
            "qf":-8.5,
            "mlp":-5.5,
            "concat":-2.5,
            "dec1":0.9,
            "lora1":3.5,
            "dec2":6.3,
            "lora2":8.9,
            "dec3":11.7,
        }

        inp=Rectangle(width=2.35,height=2.35,stroke_color=STROKE,stroke_width=1.3,fill_color="#7b7b7b",fill_opacity=0.33).move_to([x["inp"],y0,0])
        vit=make_block("ViT",2.05,1.75,fill=BLOCK,fs=21).move_to([x["vit"],y0,0])
        qf=make_block("Q-Former",2.9,1.95,fill=QF,fs=20).move_to([x["qf"],y0,0])
        mlp=make_block("MLP",1.85,1.55,fill=QF,fs=20).move_to([x["mlp"],y0,0])
        concat=make_block("Concat",1.7,2.05,fill=CONCAT,fs=18,color="#252525").move_to([x["concat"],y0,0])
        dec1=make_block("Decoder 1",2.25,2.25,fill=BLOCK,fs=19).move_to([x["dec1"],y0,0])
        lora1=make_block("LoRA",1.0,1.05,fill=LORA,fs=18).move_to([x["lora1"],y0,0])
        dec2=make_block("Decoder 2",2.25,2.25,fill=BLOCK,fs=19).move_to([x["dec2"],y0,0])
        lora2=make_block("LoRA",1.0,1.05,fill=LORA,fs=18).move_to([x["lora2"],y0,0])
        dec3=make_block("Decoder 3",2.25,2.25,fill=BLOCK,fs=19).move_to([x["dec3"],y0,0])

        self.add(inp,vit,qf,mlp,concat,dec1,lora1,dec2,lora2,dec3)

        # Input image patch grid
        gx,gy=inp.get_left()[0],inp.get_bottom()[1]
        gw,gh=inp.width,inp.height
        for k in range(1,4):
            self.add(Line([gx+gw*k/4,gy,0],[gx+gw*k/4,gy+gh,0],stroke_color=STROKE,stroke_width=0.8))
            self.add(Line([gx,gy+gh*k/4,0],[gx+gw,gy+gh*k/4,0],stroke_color=STROKE,stroke_width=0.8))
        self.add(Text("Input Image",font=FONT,font_size=18,color=TXT).next_to(inp,UP,buff=0.25))

        # Token stacks + separated labels (fixed y bands)
        vit_stack=make_stack(14,TOK_GRAY).next_to(vit,UP,buff=0.26)
        q_stack=make_stack(8,TOK_BLUE).next_to(qf,RIGHT,buff=0.36)
        img_stack=make_stack(8,TOK_BLUE).next_to(mlp,RIGHT,buff=0.36)
        txt_stack=make_stack(4,TOK_RED).next_to(concat,UP,buff=0.24)

        seq_blue=make_stack(7,TOK_BLUE)
        seq_red=make_stack(4,TOK_RED)
        seq=VGroup(seq_red,seq_blue).arrange(DOWN,buff=0.12).next_to(concat,RIGHT,buff=0.40)

        self.add(vit_stack,q_stack,img_stack,txt_stack,seq)

        self.add(
            Text("ViT Embeddings",font=FONT,font_size=16,color=TXT).next_to(vit_stack,UP,buff=0.12),
            Text("Image Query Tokens",font=FONT,font_size=16,color=TXT).move_to([q_stack.get_center()[0],1.1,0]),
            Text("Image Tokens",font=FONT,font_size=16,color=TXT).move_to([img_stack.get_center()[0],1.1,0]),
            Text("Text Tokens",font=FONT,font_size=16,color=TXT).next_to(txt_stack,RIGHT,buff=0.16),
            Text("Input Sequence",font=FONT,font_size=18,color=TXT).next_to(seq,DOWN,buff=0.30),
        )
        self.add(Arrow(txt_stack.get_bottom(),concat.get_top()+UP*0.05,buff=0.05,stroke_width=1.9,max_tip_length_to_length_ratio=0.18,color=TXT))
        self.add(Brace(seq,DOWN,color=TXT,stroke_width=1.2))

        # Straight consistent arrows.
        def connect(a,b):
            return Arrow(a.get_right()+RIGHT*0.03,b.get_left()+LEFT*0.03,buff=0.07,stroke_width=1.8,max_tip_length_to_length_ratio=0.18,color=STROKE)

        self.add(
            connect(inp,vit[0]),
            connect(vit[0],qf[0]),
            connect(qf[0],mlp[0]),
            connect(mlp[0],concat[0]),
            connect(concat[0],dec1[0]),
            connect(dec1[0],lora1[0]),
            connect(lora1[0],dec2[0]),
            connect(dec2[0],lora2[0]),
            connect(lora2[0],dec3[0]),
        )

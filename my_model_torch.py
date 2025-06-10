
import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_model import NdMamba2

class myTestModel(nn.Module):
    def __init__(self, num_class) -> None:
        super(myTestModel, self).__init__()
        # 层归一化
        self.layernorm_v = nn.LayerNorm(709)
        self.layernorm_t = nn.LayerNorm(1024)
        self.layernorm_a = nn.LayerNorm(512)
        # 投影层
        self.proj_v = nn.Linear(709, 512)
        self.proj_t = nn.Linear(1024, 512)
        
        # 可学习的 [CLS] 标记，形状为 [1, 1, 512]
        self.cls_token_a = nn.Parameter(torch.zeros(1, 1, 512))  # 音频 [CLS]
        self.cls_token_v = nn.Parameter(torch.zeros(1, 1, 512))  # 视觉 [CLS]
        self.cls_token_t = nn.Parameter(torch.zeros(1, 1, 512))  # 文本 [CLS]
        
        # Transformer 编码器层（自注意力）
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8, dropout=0.3, activation='gelu', batch_first=True)
        self.au_trans1 = nn.TransformerEncoder(encoder_layer=self.encoder_layer, num_layers=2)
        self.vi_trans1 = nn.TransformerEncoder(encoder_layer=self.encoder_layer, num_layers=2)
        self.text_trans1 = nn.TransformerEncoder(encoder_layer=self.encoder_layer, num_layers=2)
        
        # Cross-attention 层
        self.audio_cross_attn = nn.MultiheadAttention(embed_dim=512, num_heads=8, dropout=0.3, batch_first=True)
        self.vision_cross_attn = nn.MultiheadAttention(embed_dim=512, num_heads=8, dropout=0.3, batch_first=True)
        self.text_cross_attn = nn.MultiheadAttention(embed_dim=512, num_heads=8, dropout=0.3, batch_first=True)
        
        # Dropout 和层归一化
        self.dropout = nn.Dropout(0.3)
        self.norm_a = nn.LayerNorm(512)
        self.norm_v = nn.LayerNorm(512)
        self.norm_t = nn.LayerNorm(512)
        
        # 融合层（拼接三个 [CLS] 表示）
        self.mlp = nn.Linear(512 * 3, num_class)  # 修改为 512 * 3，因为融合了三个模态的 [CLS]

    def contrastive_loss(self, cls_a, cls_v, cls_t, temperature=0.5):
            # 归一化特征向量
            cls_a = F.normalize(cls_a, dim=-1)
            cls_v = F.normalize(cls_v, dim=-1)
            cls_t = F.normalize(cls_t, dim=-1)
            
            # 计算相似度矩阵 [batch_size, batch_size]
            sim_matrix1 = torch.mm(cls_a, cls_t.t()) / temperature
            sim_matrix2 = torch.mm(cls_v, cls_t.t()) / temperature
            
            # 标签：对角线元素为正样本
            labels = torch.arange(cls_a.size(0)).to(cls_a.device)
            
            # 计算 InfoNCE 损失
            criterion = nn.CrossEntropyLoss()
            loss = 0.5 *criterion(sim_matrix1, labels) + 0.5 * criterion(sim_matrix2, labels)
            return loss

    def forward(self, audio, vision, text):
        # 输入形状为 [batch_size, seq_len, feature_dim]
        if audio.shape[0] != vision.shape[0] or vision.shape[0] != text.shape[0]:
            raise ValueError("Input shapes must be the same.")
        
        batch_size = audio.shape[0]
        
        # 前处理
        audio = self.layernorm_a(audio)  # [batch_size, seq_len_a, 512]
        vision = self.layernorm_v(vision)  # [batch_size, seq_len_v, 709]
        vision = self.proj_v(vision)  # [batch_size, seq_len_v, 512]
        text = self.layernorm_t(text)  # [batch_size, seq_len_t, 1024]
        text = self.proj_t(text)  # [batch_size, seq_len_t, 512]
        
        # 将 [CLS] 标记扩展到 [batch_size, 1, 512]
        cls_token_a = self.cls_token_a.expand(batch_size, -1, -1)
        cls_token_v = self.cls_token_v.expand(batch_size, -1, -1)
        cls_token_t = self.cls_token_t.expand(batch_size, -1, -1)
        
        # 在序列开头添加 [CLS] 标记
        audio = torch.cat([cls_token_a, audio], dim=1)  # [batch_size, seq_len_a + 1, 512]
        vision = torch.cat([cls_token_v, vision], dim=1)  # [batch_size, seq_len_v + 1, 512]
        text = torch.cat([cls_token_t, text], dim=1)  # [batch_size, seq_len_t + 1, 512]
        
        # 自注意力
        audio_self = self.au_trans1(audio)  # [batch_size, seq_len_a + 1, 512]
        vision_self = self.vi_trans1(vision)  # [batch_size, seq_len_v + 1, 512]
        text_self = self.text_trans1(text)  # [batch_size, seq_len_t + 1, 512]
        
        # Cross-attention
        audio_cross = self.audio_cross_attn(audio_self, vision_self, vision_self)[0]
        vision_cross = self.vision_cross_attn(vision_self, text_self, text_self)[0]
        text_cross = self.text_cross_attn(text_self, audio_self, audio_self)[0]
        
        # 残差连接和归一化
        audio = audio_self + self.dropout(audio_cross)
        audio = self.norm_a(audio)
        vision = vision_self + self.dropout(vision_cross)
        vision = self.norm_v(vision)
        text = text_self + self.dropout(text_cross)
        text = self.norm_t(text)
        
        # 提取 [CLS] 标记的输出（第一个位置）
        cls_a = audio[:, 0, :]  # [batch_size, 512]
        cls_v = vision[:, 0, :]  # [batch_size, 512]
        cls_t = text[:, 0, :]  # [batch_size, 512]
        
        # 融合 [CLS] 表示（拼接）
        cls_fused = torch.cat([cls_a, cls_v, cls_t], dim=1)  # [batch_size, 512 * 3]
        
        # 通过 MLP 进行分类
        output = self.mlp(cls_fused)  # [batch_size, 2]
        
        
        contrast_loss = self.contrastive_loss(cls_a, cls_v, cls_t)
        return output, contrast_loss


class MambaTrans(nn.Module):
    def __init__(self, num_class) -> None:
        super(MambaTrans, self).__init__()
        # 层归一化
        self.layernorm_v = nn.LayerNorm(709)
        self.layernorm_t = nn.LayerNorm(1024)
        self.layernorm_a = nn.LayerNorm(512)
        # 投影层
        self.proj_v = nn.Linear(709, 512)
        self.proj_t = nn.Linear(1024, 512)
        
        # 可学习的 [CLS] 标记，形状为 [1, 1, 512]
        self.cls_token_a = nn.Parameter(torch.zeros(1, 1, 512))  # 音频 [CLS]
        self.cls_token_v = nn.Parameter(torch.zeros(1, 1, 512))  # 视觉 [CLS]
        self.cls_token_t = nn.Parameter(torch.zeros(1, 1, 512))  # 文本 [CLS]
        
        # Transformer 编码器层（自注意力）
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8, dropout=0.3, activation='gelu', batch_first=True)
        self.au_trans1 = nn.TransformerEncoder(encoder_layer=self.encoder_layer, num_layers=2)
        self.vi_trans1 = nn.TransformerEncoder(encoder_layer=self.encoder_layer, num_layers=2)
        self.text_trans1 = nn.TransformerEncoder(encoder_layer=self.encoder_layer, num_layers=2)
        
        # Cross-attention 层
        self.audio_cross_attn = nn.MultiheadAttention(embed_dim=512, num_heads=8, dropout=0.3, batch_first=True)
        self.vision_cross_attn = nn.MultiheadAttention(embed_dim=512, num_heads=8, dropout=0.3, batch_first=True)
        self.text_cross_attn = nn.MultiheadAttention(embed_dim=512, num_heads=8, dropout=0.3, batch_first=True)
        
        # Dropout 和层归一化
        self.dropout = nn.Dropout(0.3)
        self.norm_a = nn.LayerNorm(512)
        self.norm_v = nn.LayerNorm(512)
        self.norm_t = nn.LayerNorm(512)
        self.mamba = NdMamba2(512 * 3, 1024, 512) # 修改为 512 * 3，因为融合了三个模态的 [CLS]
        # 融合层（拼接三个 [CLS] 表示）
        self.mlp = nn.Linear(1024, num_class)  

    def contrastive_loss(self, cls_a, cls_v, cls_t, temperature=0.5):
            # 归一化特征向量
            cls_a = F.normalize(cls_a, dim=-1)
            cls_v = F.normalize(cls_v, dim=-1)
            cls_t = F.normalize(cls_t, dim=-1)
            
            # 计算相似度矩阵 [batch_size, batch_size]
            sim_matrix1 = torch.mm(cls_a, cls_t.t()) / temperature
            sim_matrix2 = torch.mm(cls_v, cls_t.t()) / temperature
            
            # 标签：对角线元素为正样本
            labels = torch.arange(cls_a.size(0)).to(cls_a.device)
            
            # 计算 InfoNCE 损失
            criterion = nn.CrossEntropyLoss()
            loss = 0.5 *criterion(sim_matrix1, labels) + 0.5 * criterion(sim_matrix2, labels)
            return loss

    def forward(self, audio, vision, text):
        # 输入形状为 [batch_size, seq_len, feature_dim]
        if audio.shape[0] != vision.shape[0] or vision.shape[0] != text.shape[0]:
            raise ValueError("Input shapes must be the same.")
        
        batch_size = audio.shape[0]
        
        # 前处理
        audio = self.layernorm_a(audio)  # [batch_size, seq_len_a, 512]
        vision = self.layernorm_v(vision)  # [batch_size, seq_len_v, 709]
        vision = self.proj_v(vision)  # [batch_size, seq_len_v, 512]
        text = self.layernorm_t(text)  # [batch_size, seq_len_t, 1024]
        text = self.proj_t(text)  # [batch_size, seq_len_t, 512]
        
        # 将 [CLS] 标记扩展到 [batch_size, 1, 512]
        cls_token_a = self.cls_token_a.expand(batch_size, -1, -1)
        cls_token_v = self.cls_token_v.expand(batch_size, -1, -1)
        cls_token_t = self.cls_token_t.expand(batch_size, -1, -1)
        
        # 在序列开头添加 [CLS] 标记
        audio = torch.cat([cls_token_a, audio], dim=1)  # [batch_size, seq_len_a + 1, 512]
        vision = torch.cat([cls_token_v, vision], dim=1)  # [batch_size, seq_len_v + 1, 512]
        text = torch.cat([cls_token_t, text], dim=1)  # [batch_size, seq_len_t + 1, 512]
        
        # 自注意力
        audio_self = self.au_trans1(audio)  # [batch_size, seq_len_a + 1, 512]
        vision_self = self.vi_trans1(vision)  # [batch_size, seq_len_v + 1, 512]
        text_self = self.text_trans1(text)  # [batch_size, seq_len_t + 1, 512]
        
        # Cross-attention
        audio_cross = self.audio_cross_attn(audio_self, vision_self, vision_self)[0]
        vision_cross = self.vision_cross_attn(vision_self, text_self, text_self)[0]
        text_cross = self.text_cross_attn(text_self, audio_self, audio_self)[0]
        
        # 残差连接和归一化
        audio = audio_self + self.dropout(audio_cross)
        audio = self.norm_a(audio)
        vision = vision_self + self.dropout(vision_cross)
        vision = self.norm_v(vision)
        text = text_self + self.dropout(text_cross)
        text = self.norm_t(text)
        
        # 提取 [CLS] 标记的输出（第一个位置）
        cls_a = audio[:, 0, :]  # [batch_size, 512]
        cls_v = vision[:, 0, :]  # [batch_size, 512]
        cls_t = text[:, 0, :]  # [batch_size, 512]
        
        # 融合 [CLS] 表示（拼接）
        cls_fused = torch.cat([cls_a, cls_v, cls_t], dim=1)  # [batch_size, 512 * 3]
        cls_fused = torch.unsqueeze(cls_fused, 1).transpose(1, 2)
        
        cls_mamba = self.mamba(cls_fused)
        cls_mamba = cls_mamba.transpose(1, 2).squeeze(1)
        # 通过 MLP 进行分类
        output = self.mlp(cls_mamba)  # [batch_size, 2]
        
        
        contrast_loss = self.contrastive_loss(cls_a, cls_v, cls_t)
        return output, contrast_loss

    
if __name__ == '__main__':
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    audio = torch.randn((3, 16, 512))
    vision = torch.randn((3, 16, 709))
    text = torch.randn((3, 16, 1024))

    model = MambaTrans(num_class=2)
    out = model(audio, vision, text)
    print(out)



    


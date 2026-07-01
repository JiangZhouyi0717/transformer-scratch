import torch
def scaled_dot_product_attention(Q,K,V):
    """

    :param Q: (batch_size, sequence_length, d_k)
    :param K: (batch_size, sequence_length, d_k)
    :param V: (batch_size, sequence_length, d_k)
    :return:
    output: (batch_size, sequence_length, d_k)
    weights: (batch_size, sequence_length, sequence_length) 注意力权重矩阵
    """
    d_k=K.shape[-1]
    scores=Q@K.transpose(-2,-1)
    scaled_scores=scores/(d_k**0.5)
    weights=torch.softmax(scaled_scores,dim=-1)
    output=weights@V
    return output, weights

"============测试代码============"
if __name__=="__main__":
    print("=== 测试 Scaled Dot-Product Attention ===")
    #设置参数
    batch_size=2
    seq_len=4
    d_k=8

    Q=torch.randn(batch_size, seq_len, d_k,device='cuda')
    K=torch.randn(batch_size, seq_len, d_k,device='cuda')
    V=torch.randn(batch_size, seq_len, d_k,device='cuda')
    output, weights = scaled_dot_product_attention(Q,K,V)

    print(f"Q形状：{Q.shape}")
    print(f"K形状：{K.shape}")
    print(f"V形状：{V.shape}")
    print(f"output形状：{output.shape}")
    print(f"weights形状：{weights.shape}")
    row_sum=weights[0,0,:].sum()
    print(f"第 1 个样本第 1 行的权重和: {row_sum.item():.4f}")
    print("\n✅ Attention 测试通过！")

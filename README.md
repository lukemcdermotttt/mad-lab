# Parametric Attention

This repository is based on Mad-lab. To run this code, please follow the environment instructions on their repository.

See `mad/model/layers/mlp_attention_simple.py` for our implementation of Parametric Attention. 

The config for this layer is located at `configs/layrs/simple-mlp-attention.yml`.

## Psuedo-Code for Parametric Attention
```python
class SimpleMLPAttention(nn.Module):
    def init():
        ...

    def forward(hidden_states: torch.Tensor):

            W_in, W_out = init_MLP_w_trainable_parameter()
            q,k,v = generate_qkv(hidden_states, use_conv=True, use_rope=True)
            
            for chunk in sequence:
                #Online Forward Pass (query attends to sliding window attention and online MLP parameters)
                output[chunk] = FlashAttention(query  = q_chunk, 
                                               keys   = [W_in, k_chunk],
                                               values = [W_out, v_chunk],
                                               causal = True)

                #Online Backward Pass (update MLP parameters with the chunk of kv-pairs before evicting them)
                Grad_in = FlashAttention(W_out, v_chunk, k_chunk, causal=False) #optionally: fuse all FA calls
                Grad_out = FlashAttention(W_in, k_chunk, v_chunk, causal=False) 
                W_in = W_in - lr_in * Grad_in #optionally: use a momentum state here instead.
                W_out = W_out - lr_out * Grad_out

            #Use output gating from Atlas
            output = projection_layer(layer_norm(output) * gate_layer(hidden_states))
            return output
```
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


# Notes about the MAD-Lab Procedure
## The MAD synthetic tasks
MAD spans six simple token manipulation tasks. We provide a brief overview of each task in the following. For more details, please see our paper.

### `in-context-recall`
<img src="./assets/recall.png" alt="recall" width="300"/>
To answer a prompt well, language models must be able to understand and learn from new information presented in the prompt (so-called in-context learning). A wealth of empirical work has demonstrated that the associative recall task is well-suited to test the in-context learning skill. MAD uses a multi-query variant of this task: Given an input sequence of key-value pairs, models are tasked with retrieving all values from the input sequence associated with keys that were already shown in the input sequence. Note that while the mapping from keys to values is consistent within an input sequence, it is randomly shuffled between sequences.

### `fuzzy-in-context-recall`
<img src="./assets/fuzzy_recall.png" alt="fuzzy_recall" width="300"/>
In language, semantic units are often spread out over multiple adjacent tokens (e.g., "blue sky" vs "gray sky"). To test how capable a model is of semantically grouping together adjacent tokens, MAD utilizes a variant of in-context recall, in which keys and values are composed of a variable number of adjacent tokens. Specifically, for each sequence, variable length keys and values are randomly drawn from the vocabulary and then assigned into pairs. Since the structure of key/value lengths in a sequence, as well as the mapping from keys to values, change between sequences, fuzzy recall can be treated as a more challenging variant of in-context recall.

### `noisy-in-context-recall`
<img src="./assets/noisy_recall.png" alt="noisy_recall" width="300"/>
To answer a prompt well, language models must be able to ignore irrelevant information of the input. To test this skill, MAD uses another adaptation of in-context recall, in which irrelevant information, represented by tokens from a distinct vocabulary, is added in an arbitrary and variable pattern in between the key-value pairs.
Note that this adds a memorization aspect to the task, as models need to learn during training to ignore tokens from the noise vocabulary.

### `selective-copying`
<img src="./assets/selective_copy.png" alt="selective_copy" width="300"/>
In addition to ignoring irrelevant information of an input, language models must be able to selectively remember relevant information of an input. To test this skill, MAD uses a selective copying task, in which models are tasked with copying tokens from one position of an input sequence to a later position of the sequence, while ignoring irrelevant noise tokens that are randomly inserted into the sequence. Importantly, tokens are always copied in their order of occurrence. Models thereby need to not just remember the tokens that are to be copied but also their specific order of occurrence in the sequence.

### `compression`
<img src="./assets/compression.png" alt="compression" width="300"/>
Recent findings in the mechanistic interpretability literature indicate that a key skill of language models is "token concatenation", where early attention layers assemble information that is spread across multiple tokens in an input onto another token so that the assembled information can then be decoded well by subsequent MLPs. To test the ability of a model to perform token concatenation, even without attention and MLP, MAD utilizes a compression task. In this task, models are trained to compress a random sequence of input tokens into a single aggregation token so that the original input sequence can be fully recovered from the aggregation token by a subsequent MLP.

### `memorization`
<img src="./assets/memorization.png" alt="memorization" width="300"/>
In addition to manipulating and retrieving information from an input sequence, language modeling requires the memorization of factual knowledge. To test this skill, MAD utilizes a memorization task, in which models are tasked with learning a fixed key-value mapping (resembling facts in language) from the training data. Unlike recall, the mapping requires no in-context computation as the ground-truth mapping is constant across samples. 



## The MAD Protocol
MAD follows a two-step procedure, starting from the design of a new candidate architecture, followed by its systematic evaluation according to the following key principles: 

1. For each synthetic task, a MAD score is obtained by averaging architecture performances across a range of task difficulty levels. To manipulate difficulty, MAD independently varies a set of relevant experimental variables: length of the input sequence, size of the vocabulary, and size of the training set. Some tasks have additional variables such as the ratio of noise tokens in the noisy recall and selective copying tasks. For an overview of the changes applied to each task, see the changes entry in each task config in [configs/tasks/](configs/tasks/).

2. Fixed-state architectures need to be normalized to an iso-state and iso-parameter setting, including models featuring sparsely activated layers such as Mixture-of-Experts. For details on this, please see our paper!

3. To ensure that model performance estimates are not dependent on a specific optimization setting, MAD sweeps each architecture in each task setting over a 3 x 2 grid of learning rate and weight decay values (learning rates: $0.0001, 0.0005, 0.001$, weight decays: $0., 0.1$). MAD scores are based on the best runs from this sweep.

4. Model performances are always evaluated in an independent evaluation dataset, specific to each task setting.



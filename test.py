import random
import numpy as np

def generate_chains(num_chains: int, num_hops: int, is_icl: bool = False, vocab_size = 26):
    vocab = [chr(i) for i in range(65, 65 + vocab_size)]
    vars_all = []
    k = 5 if not is_icl else 3
    num_hops = num_hops if not is_icl else min(10, num_hops)
    vars_all = [
        "".join(random.choices(vocab, k=k)).upper()
        for _ in range((num_hops + 1) * num_chains)
    ]
    while len(set(vars_all)) < num_chains * (num_hops + 1):
        vars_all.append("".join(random.choices(vocab, k=k)).upper())

    vars_ret = []
    chains_ret = []
    for i in range(0, len(vars_all), num_hops + 1):
        this_vars = vars_all[i : i + num_hops + 1]
        vars_ret.append(this_vars)
        this_chain = [f"VAR {this_vars[0]} = {np.random.randint(10000, 99999)}"]
        for j in range(num_hops):
            this_chain.append(f"VAR {this_vars[j + 1]} = VAR {this_vars[j]} ")
        chains_ret.append(this_chain)
    return vars_ret, chains_ret

vars_ret, chains_ret = generate_chains(2, 3)
print(vars_ret)
print(chains_ret)
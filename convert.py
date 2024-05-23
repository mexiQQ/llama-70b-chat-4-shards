import argparse
import gc
import json
import os
import shutil
import warnings
import torch

def read_json(path):
    with open(path, "r") as f:
        return json.load(f)


def write_json(text, path):
    with open(path, "w") as f:
        json.dump(text, f)

def copy_json(input_base_path, output_llama_dir):
    ori_params_path = os.path.join(input_base_path, "params.json")
    out_params_path = os.path.join(output_llama_dir, "params.json")
    shutil.copyfile(ori_params_path, out_params_path)

def convert_to_llama_70b_1(input_base_path, llama_version, num_shards=8):
    params = read_json(os.path.join(input_base_path, "params.json"))
    n_layers = params["n_layers"]
    # n_layers = 2
    n_heads = params["n_heads"]
    n_heads_per_shard = n_heads // num_shards
    dim = params["dim"]
    dims_per_head = dim // n_heads
    
    base = params.get("rope_theta", 10000.0)
    inv_freq = 1.0 / (base ** (torch.arange(0, dims_per_head, 2).float() / dims_per_head))
    if base > 10000.0:
        max_position_embeddings = 16384
    else:
        max_position_embeddings = 2048

    vocab_size = 32000

    if "n_kv_heads" in params:
        num_key_value_heads = params["n_kv_heads"]  # for GQA / MQA
        num_local_key_value_heads = n_heads_per_shard // num_key_value_heads
        key_value_dim = dim // num_key_value_heads
    else:  # compatibility with other checkpoints
        num_key_value_heads = n_heads
        num_local_key_value_heads = n_heads_per_shard
        key_value_dim = dim

    # permute for sliced rotary
    def permute(w, n_heads=n_heads, dim1=dim, dim2=dim):
        return w.view(n_heads, dim1 // n_heads // 2, 2, dim2).transpose(1, 2).reshape(dim1, dim2)

    print(f"Fetching all parameters from the checkpoint at {input_base_path}.")
    # Sharded
    loaded = [
        torch.load(os.path.join(input_base_path, f"consolidated.{i:02d}.pth"), map_location="cpu")
        for i in range(num_shards)
    ]

    new_weights = {}
    for layer_i in range(n_layers):        
        # Sharded
        # Note that attention.w{q,k,v,o}, feed_fordward.w[1,2,3], attention_norm.weight and ffn_norm.weight share
        # the same storage object, saving attention_norm and ffn_norm will save other weights too, which is
        # redundant as other weights will be stitched from multiple shards. To avoid that, they are cloned.

        new_weights[f"layers.{layer_i}.attention.wq.weight"] = torch.cat(
            [
                loaded[i][f"layers.{layer_i}.attention.wq.weight"].view(n_heads_per_shard, dims_per_head, dim)
                for i in range(num_shards)
            ],
            dim=0,
        )

        new_weights[f"layers.{layer_i}.attention.wk.weight"] = torch.cat(
            [
                loaded[i][f"layers.{layer_i}.attention.wk.weight"].view(
                    num_local_key_value_heads, dims_per_head, dim
                )
                for i in range(num_shards)
            ],
            dim=0,
        )

        new_weights[f"layers.{layer_i}.attention.wv.weight"] = torch.cat(
            [
                loaded[i][f"layers.{layer_i}.attention.wv.weight"].view(
                    num_local_key_value_heads, dims_per_head, dim
                )
                for i in range(num_shards)
            ],
            dim=0,
        )

        new_weights[f"layers.{layer_i}.attention.wo.weight"] = torch.cat(
            [loaded[i][f"layers.{layer_i}.attention.wo.weight"] for i in range(num_shards)], dim=1
        )

        new_weights[f"layers.{layer_i}.feed_forward.w1.weight"] = torch.cat(
            [loaded[i][f"layers.{layer_i}.feed_forward.w1.weight"] for i in range(num_shards)], dim=0
        )

        new_weights[f"layers.{layer_i}.feed_forward.w2.weight"] = torch.cat(
            [loaded[i][f"layers.{layer_i}.feed_forward.w2.weight"] for i in range(num_shards)], dim=1
        )

        new_weights[f"layers.{layer_i}.feed_forward.w3.weight"] = torch.cat(
            [loaded[i][f"layers.{layer_i}.feed_forward.w3.weight"] for i in range(num_shards)], dim=0
        )

        new_weights[f"layers.{layer_i}.attention_norm.weight"] = loaded[0][
            f"layers.{layer_i}.attention_norm.weight"
        ]

        new_weights[f"layers.{layer_i}.ffn_norm.weight"] = loaded[0][
            f"layers.{layer_i}.ffn_norm.weight"
        ]

    concat_dim = 0 if llama_version == 3 else 1
    new_weights["tok_embeddings.weight"] = torch.cat(
        [loaded[i]["tok_embeddings.weight"] for i in range(num_shards)], dim=concat_dim
    )
    new_weights["norm.weight"] = loaded[0]["norm.weight"]
    new_weights["output.weight"] = torch.cat([loaded[i]["output.weight"] for i in range(num_shards)], dim=0)

    # Make space so we can load the model properly now.
    del loaded
    gc.collect()

    return new_weights

def convert_to_llama_70b_2(state_dict, input_base_path, output_llama_dir, num_shards=4):

    params = read_json(os.path.join(input_base_path, "params.json"))
    n_layers = params["n_layers"]
    # n_layers = 2
    n_heads = params["n_heads"]
    n_heads_per_shard = n_heads // num_shards
    dim = params["dim"]
    dims_per_head = dim // n_heads

    base = params.get("rope_theta", 10000.0)
    inv_freq = 1.0 / (base ** (torch.arange(0, dims_per_head, 2).float() / dims_per_head))
    if base > 10000.0:
        max_position_embeddings = 16384
    else:
        max_position_embeddings = 2048

    if "n_kv_heads" in params:
        num_key_value_heads = params["n_kv_heads"]  # for GQA / MQA
        num_local_key_value_heads = n_heads_per_shard // num_key_value_heads
        key_value_dim = dim // num_key_value_heads
    else:
        num_key_value_heads = n_heads
        num_local_key_value_heads = n_heads_per_shard
        key_value_dim = dim

    new_weights = {}
    for layer_i in range(n_layers):        
        new_weights[f"layers.{layer_i}.attention.wq.weight"] = torch.chunk(
            state_dict[f"layers.{layer_i}.attention.wq.weight"], num_shards, dim=0
        )
        
        new_weights[f"layers.{layer_i}.attention.wk.weight"] = torch.chunk(
            state_dict[f"layers.{layer_i}.attention.wk.weight"], num_shards, dim=0
        )

        new_weights[f"layers.{layer_i}.attention.wv.weight"] = torch.chunk(
            state_dict[f"layers.{layer_i}.attention.wv.weight"], num_shards, dim=0
        )

        # The other weights remain unchanged
        new_weights[f"layers.{layer_i}.attention.wo.weight"] = torch.chunk(
            state_dict[f"layers.{layer_i}.attention.wo.weight"], num_shards, dim=1
        )

        new_weights[f"layers.{layer_i}.feed_forward.w1.weight"] = torch.chunk(
            state_dict[f"layers.{layer_i}.feed_forward.w1.weight"], num_shards, dim=0
        )

        new_weights[f"layers.{layer_i}.feed_forward.w2.weight"] = torch.chunk(
            state_dict[f"layers.{layer_i}.feed_forward.w2.weight"], num_shards, dim=1
        )

        new_weights[f"layers.{layer_i}.feed_forward.w3.weight"] = torch.chunk(
            state_dict[f"layers.{layer_i}.feed_forward.w3.weight"], num_shards, dim=0
        )

        new_weights[f"layers.{layer_i}.attention_norm.weight"] = state_dict[f"layers.{layer_i}.attention_norm.weight"].clone()

        new_weights[f"layers.{layer_i}.ffn_norm.weight"] = state_dict[f"layers.{layer_i}.ffn_norm.weight"].clone()

    # Handle the embeddings and output head weights
    new_weights["tok_embeddings.weight"] = torch.chunk(
        state_dict[f"tok_embeddings.weight"], num_shards, dim=1
    )
    new_weights["norm.weight"] = state_dict["norm.weight"]
    new_weights["output.weight"] = torch.chunk(
        state_dict[f"output.weight"], num_shards, dim=0
    )

    # Shard the weights
    weight_shards = [{} for _ in range(num_shards)]
    for key, value in new_weights.items():
        if "norm" in key:
            for i in range(num_shards):
                weight_shards[i][key] = value.clone()

        if "wk" in key or "wq" in key or "wv" in key:
            for i in range(num_shards):
                weight_shards[i][key] = value[i].view(-1, dim).clone()

        if "wo" in key or "w1" in key or "w2" in key or "w3" in key or "tok_embeddings" in key or "output" in key:
            for i in range(num_shards):
                weight_shards[i][key] = value[i].clone()

    for i in range(num_shards):
        weight_shards[i]["rope.freqs"] = inv_freq.to(torch.bfloat16).clone()
        
    # Save the sharded weights in the expected format
    for i, shard in enumerate(weight_shards):
        path = os.path.join(output_llama_dir, f"consolidated.0{i}.pth")
        torch.save(shard, path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_llama_path", help="Path to input llama.")
    parser.add_argument("--input_shards", type=int, default=8)
    parser.add_argument("--output_llama_path", help="Path to save the converted LLaMA model.")
    parser.add_argument("--output_shards", type=int, default=4)
    parser.add_argument("--llama_version", type=int, default=3)
    args = parser.parse_args()
    os.makedirs(args.output_llama_path, exist_ok=True)

    copy_json(args.input_llama_path, args.output_llama_path)

    state_dict1 = convert_to_llama_70b_1(
        args.input_llama_path,
        args.llama_version,
        num_shards = args.input_shards
    )

    # state_dict2 = convert_to_llama_70b_1(
    #     args.output_llama_path,
    #     num_shards = args.output_shards
    # )

    # for key in state_dict1.keys():
    #     val1 = state_dict1[key]
    #     val2 = state_dict2[key]
    #     print(key, torch.all(val1==val2))
    # import pdb; pdb.set_trace()

    convert_to_llama_70b_2(
        state_dict1,
        args.input_llama_path,
        args.output_llama_path,
        num_shards = args.output_shards
    )

if __name__ == "__main__":
    main()

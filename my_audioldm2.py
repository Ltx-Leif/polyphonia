import torch
import torch.nn.functional as F
from diffusers import AudioLDM2Pipeline


def load_audioldm2(precision_t=torch.float16, device="cuda"):
    repo_id = "cvssp/audioldm2"
    pipe = AudioLDM2Pipeline.from_pretrained(repo_id, torch_dtype=precision_t)
    pipe = pipe.to(device)
    vae = pipe.vae
    tokenizer = pipe.tokenizer
    tokenizer_2 = pipe.tokenizer_2
    text_encoder = pipe.text_encoder
    text_encoder_2 = pipe.text_encoder_2
    vocoder = pipe.vocoder
    projection_model = pipe.projection_model
    language_model = pipe.language_model
    unet = pipe.unet
    scheduler = pipe.scheduler
    del pipe

    return vae, tokenizer, tokenizer_2, text_encoder, text_encoder_2, vocoder, projection_model, language_model, unet, scheduler


def load_tokenizer_2(precision_t=torch.float16, device="cuda"):
    repo_id = "cvssp/audioldm2"
    pipe = AudioLDM2Pipeline.from_pretrained(repo_id, torch_dtype=precision_t)
    pipe = pipe.to(device)
    tokenizer_2 = pipe.tokenizer_2
    del pipe

    return tokenizer_2


def decode_latent(latents, vae):
    # scale and decode the image latents with vae
    latents = 1 / vae.config.scaling_factor * latents
    with torch.no_grad():
        mel_spectrogram = vae.decode(latents).sample
    return mel_spectrogram


def encode_latent(images, vae):
    # encode the image with vae
    with torch.no_grad():
        latents = vae.encode(images).latent_dist.mode()
    latents = vae.config.scaling_factor * latents
    return latents


def get_text_embedding(text, text_encoder, tokenizer, device="cuda"):
    # TODO currently, hard-coding for stable diffusion
    with torch.no_grad():

        prompt = [text]
        batch_size = len(prompt)
        text_input = tokenizer(prompt, padding="max_length",
                               max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")

        text_embeddings = text_encoder(
            text_input.input_ids.to(device))[0].to(device)
        max_length = text_input.input_ids.shape[-1]
        # print(max_length, text_input.input_ids)
        uncond_input = tokenizer(
            [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
        )
        uncond_embeddings = text_encoder(
            uncond_input.input_ids.to(device))[0].to(device)

    return text_embeddings, uncond_embeddings


def get_attn_layers(unet):
    names = ['down', 'up', 'mid']
    block_idxs = dict(zip(names, [list(range(1, 4)), list(range(3)), []]))
    layer_idxs = dict(
        zip(names, [list(range(6)), list(range(9)), list(range(3))]))
    attn_layers = dict.fromkeys(names)

    def _tag_branch(attn_module, branch: str):
        """
        为交叉注意力模块打上分支标签。
        """
        try:
            setattr(attn_module, "_melodia_branch", branch)
            for tb in getattr(attn_module, "transformer_blocks", []):
                if hasattr(tb, "attn2"):
                    setattr(tb.attn2, "_melodia_branch", branch)
        except Exception:
            pass

    for name in names:
        attn_layers[name] = []
        if len(block_idxs[name]) > 0:
            for idx in block_idxs[name]:
                block = getattr(unet, f'{name}_blocks')[idx]
                num_cross = len(getattr(block, "cross_attention_dim", [])) or 1
                for layer_idx in layer_idxs[name]:
                    attn = block.attentions[layer_idx]
                    branch_idx = layer_idx % num_cross
                    branch_tag = "gpt" if branch_idx <= 1 else "text"
                    _tag_branch(attn, branch_tag)
                    attn_layers[name].append(attn)
        else:
            block = getattr(unet, f'{name}_block')
            num_cross = len(getattr(block, "cross_attention_dim", [])) or 1
            for layer_idx in layer_idxs[name]:
                attn = block.attentions[layer_idx]
                branch_idx = layer_idx % num_cross
                branch_tag = "gpt" if branch_idx <= 1 else "text"
                _tag_branch(attn, branch_tag)
                attn_layers[name].append(attn)

    return attn_layers


# Diffusers attention code for getting query, key, value and attention map
def attention_op(attn, hidden_states, encoder_hidden_states=None, attention_mask=None, query=None, key=None, value=None, attention_probs=None, temperature=1.0):
    residual = hidden_states

    # if attn.spatial_norm is not None:
    #     hidden_states = attn.spatial_norm(hidden_states, temb)

    input_ndim = hidden_states.ndim

    if input_ndim == 4:
        batch_size, channel, height, width = hidden_states.shape
        hidden_states = hidden_states.view(
            batch_size, channel, height * width).transpose(1, 2)

    # if key is None:
    batch_size, sequence_length, _ = (
        hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
    )
    # else:
    #     batch_size, sequence_length, _ = key.shape
    #     batch_size /= 8
    #     _ *= 8

    attention_mask = attn.prepare_attention_mask(
        attention_mask, sequence_length, batch_size)

    if attn.group_norm is not None:
        hidden_states = attn.group_norm(
            hidden_states.transpose(1, 2)).transpose(1, 2)

    if attention_probs is not None:
        if value is None:
            value = attn.to_v(encoder_hidden_states)
            value = attn.head_to_batch_dim(value)
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(
                -1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        # 获取注意力模块的scale值，使用getattr确保健壮性
        scale = getattr(attn, 'scale', 1.0 / (attn.heads ** 0.5))

        return attention_probs, query, key, value, hidden_states, scale

    if query is None:
        query = attn.to_q(hidden_states)
        query = attn.head_to_batch_dim(query)

    if encoder_hidden_states is None:
        encoder_hidden_states = hidden_states
    elif attn.norm_cross:
        encoder_hidden_states = attn.norm_encoder_hidden_states(
            encoder_hidden_states)

    if key is None:
        key = attn.to_k(encoder_hidden_states)
        key = attn.head_to_batch_dim(key)
    if value is None:
        value = attn.to_v(encoder_hidden_states)
        value = attn.head_to_batch_dim(value)

    if key.shape[0] != query.shape[0]:
        key, value = key[:query.shape[0]], value[:query.shape[0]]

    if key.shape[1] < value.shape[1]:
        key = F.pad(key, (0, 0, 0, value.shape[1]-key.shape[1], 0, 0))
    elif key.shape[1] > value.shape[1]:
        key = key[:, :value.shape[1], :]

    # apply temperature scaling
    query = query * temperature  # same as applying it on qk matrix

    attention_probs = attn.get_attention_scores(query, key, attention_mask)

    batch_heads, img_len, txt_len = attention_probs.shape

    # h = w = int(img_len ** 0.5)
    # attention_probs_return = attention_probs.reshape(batch_heads // attn.heads, attn.heads, h, w, txt_len)

    hidden_states = torch.bmm(attention_probs, value)
    hidden_states = attn.batch_to_head_dim(hidden_states)

    # linear proj
    hidden_states = attn.to_out[0](hidden_states)
    # dropout
    hidden_states = attn.to_out[1](hidden_states)

    if input_ndim == 4:
        hidden_states = hidden_states.transpose(
            -1, -2).reshape(batch_size, channel, height, width)

    if attn.residual_connection:
        hidden_states = hidden_states + residual

    hidden_states = hidden_states / attn.rescale_output_factor

    # 获取注意力模块的scale值，使用getattr确保健壮性
    scale = getattr(attn, 'scale', 1.0 / (attn.heads ** 0.5))

    return attention_probs, query, key, value, hidden_states, scale


if __name__ == "__main__":
    vae, tokenizer, tokenizer_2, text_encoder, text_encoder_2, vocoder, projection_model, language_model, unet, scheduler = load_audioldm2()
    resnet, attn = get_attn_layers(unet)

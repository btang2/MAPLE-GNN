import pathlib
import torch
import pandas as pd
from esm import FastaBatchedDataset, pretrained

def extract_embeddings(model_name, fasta_file, output_dir, tokens_per_batch=4096, seq_length=3000, repr_layers=[6]):
    model, alphabet = pretrained.load_model_and_alphabet(model_name)
    model.eval()

    if torch.cuda.is_available():
        model = model.cuda()

    dataset = FastaBatchedDataset.from_file(fasta_file)
    batches = dataset.get_batch_indices(tokens_per_batch, extra_toks_per_seq=1)

    data_loader = torch.utils.data.DataLoader(
        dataset,
        collate_fn=alphabet.get_batch_converter(seq_length),
        batch_sampler=batches
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for batch_idx, (labels, strs, toks) in enumerate(data_loader):
            print(f'Processing batch {batch_idx + 1} of {len(batches)}')

            if torch.cuda.is_available():
                toks = toks.to(device="cuda", non_blocking=True)

            out = model(toks, repr_layers=repr_layers, return_contacts=False)

            logits = out["logits"].to(device="cpu")
            representations = {layer: t.to(device="cpu") for layer, t in out["representations"].items()}

            for i, label in enumerate(labels):
                entry_id = label.split()[0]

                filename = output_dir / f"{entry_id}.pt"
                truncate_len = min(seq_length, len(strs[i]))

                result = {"entry_id": entry_id}
                result["representations"] = {
                    layer: t[i, 1: truncate_len + 1].clone()
                    for layer, t in representations.items()
                }

                torch.save(result, filename)

model_name = 'esm2_t6_8M_UR50D' #esm2_t33_650M_UR50D, esm2_t6_8M_UR50D
fasta_file = pathlib.Path('Fasta/test0918.fasta')
output_dir = pathlib.Path('Fasta/test0918')

extract_embeddings(model_name, fasta_file, output_dir)

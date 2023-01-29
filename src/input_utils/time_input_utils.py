import os
import json
import torch

from tqdm import tqdm


def fft_preprocess(time_input, args):
    """Run FFT on the time-domain input.
    time_input: [b, c, i, s]
    freq_output: [b, c, i, s]
    """
    freq_output = dict()

    for loc in time_input:
        freq_output[loc] = dict()
        for mod in time_input[loc]:
            loc_mod_freq_output = torch.fft.fft(time_input[loc][mod], dim=-1)
            loc_mod_freq_output = torch.view_as_real(loc_mod_freq_output)
            loc_mod_freq_output = loc_mod_freq_output.permute(0, 1, 4, 2, 3)
            b, c1, c2, i, s = loc_mod_freq_output.shape
            loc_mod_freq_output = loc_mod_freq_output.reshape(b, c1 * c2, i, s)
            freq_output[loc][mod] = loc_mod_freq_output

    return freq_output


def count_range(args, data_loader):
    """Count the data range for each modality."""
    loc_mod_max_abs = {}
    for i, (loc_inputs, labels) in tqdm(enumerate(data_loader), total=len(data_loader)):
        for loc in loc_inputs:
            for mod in loc_inputs[loc]:
                if mod not in loc_mod_max_abs:
                    loc_mod_max_abs[mod] = torch.max(torch.abs(loc_inputs[loc][mod])).item()
                else:
                    loc_mod_max_abs[mod] = max(
                        loc_mod_max_abs[mod],
                        torch.max(torch.abs(loc_inputs[loc][mod])).item(),
                    )

    log_file = os.path.join(args.log_path, f"value_range.json")

    # load existing file
    if os.path.exists(log_file):
        value_range_cache = json.load(open(log_file, "r"))
    else:
        value_range_cache = {}

    # save the new value range
    value_range_cache["time"] = loc_mod_max_abs
    with open(log_file, "w") as f:
        f.write(json.dumps(value_range_cache, indent=4))

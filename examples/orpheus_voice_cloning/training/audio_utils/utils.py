"""Utility functions for audio processing and tokenization for Orpheus voice cloning training."""
import torch
from torchaudio import transforms as T

from snac import SNAC

CODES_LIST_NAME = "codes_list"

## specific to Orpheus tokenizer
TOKENIZER_LENGTH = 128256
START_OF_TEXT = 128000
END_OF_TEXT = 128009

START_OF_SPEECH = TOKENIZER_LENGTH + 1
END_OF_SPEECH = TOKENIZER_LENGTH + 2

START_OF_HUMAN = TOKENIZER_LENGTH + 3
END_OF_HUMAN = TOKENIZER_LENGTH + 4

START_OF_AI = TOKENIZER_LENGTH + 5
END_OF_AI =  TOKENIZER_LENGTH + 6
PAD_TOKEN = TOKENIZER_LENGTH + 7

AUDIO_TOKENS_START = TOKENIZER_LENGTH + 10

def load_snac_model(model_name="hubertsiuzdak/snac_24khz"):
    """Multi-Scale Neural Audio Codec (SNAC) compressess audio 
    into discrete codes at a low bitrate.

    SNAC encodes audio into multiple hierarchical levels with 
    different temporal resolutions:
        codes[0]: Coarsest level (lowest frequency, most compressed)
        codes[1]: Middle level (2x higher frequency than level 0)
        codes[2]: Finest level (4x higher frequency than level 0)
    """

    snac_model = SNAC.from_pretrained(model_name)
    snac_model = snac_model.to("cuda")
    return snac_model

# Tokenization functions
def tokenise_audio(
    waveform, 
    orig_freq=24000, 
    new_freq=24000,
    snac_model=None,
    ):
    waveform = torch.from_numpy(waveform).unsqueeze(0)
    waveform = waveform.to(dtype=torch.float32)
    resample_transform = T.Resample(orig_freq=orig_freq, new_freq=new_freq)
    waveform = resample_transform(waveform)

    waveform = waveform.unsqueeze(0).to("cuda")

    #generate the codes from snac
    if snac_model is None:
        snac_model = load_snac_model()

    with torch.inference_mode():
        codes = snac_model.encode(waveform)

    # We need to convert heirarchical codes into a single flat sequence
    # that encodes all hierarchical levels interleaved in a specific pattern 
    # and unique token IDs for each level and temporal position.
    all_codes = []
    for i in range(codes[0].shape[1]):
        all_codes.append(codes[0][0][i].item()+128266) # Level 0 - base offset
        all_codes.append(codes[1][0][2*i].item()+128266+4096)
        all_codes.append(codes[2][0][4*i].item()+128266+(2*4096))
        all_codes.append(codes[2][0][(4*i)+1].item()+128266+(3*4096))
        all_codes.append(codes[1][0][(2*i)+1].item()+128266+(4*4096))
        all_codes.append(codes[2][0][(4*i)+2].item()+128266+(5*4096))
        all_codes.append(codes[2][0][(4*i)+3].item()+128266+(6*4096))

    return all_codes


def add_codes(example, audio_field="audio"):
    codes_list = None
    try:
        answer_audio = example.get(audio_field)
        # If there's a valid audio array, tokenise it
        if answer_audio and "array" in answer_audio:
            audio_array = answer_audio["array"]
            codes_list = tokenise_audio(audio_array)
    except Exception as e:
        print(f"Skipping row due to error: {e}")
        # Keep codes_list as None if we fail 
        # TODO: continue instead? see how many rows fail
    example[CODES_LIST_NAME] = codes_list

    return example

def remove_duplicate_frames(example): # called in dataset map 

    # After flattening the codec from SNAC, the output should be 
    # divisible by 7. (1 token from level 0, 2 from level 1, 4 from level 2)

    vals = example[CODES_LIST_NAME]
    if len(vals) % 7 != 0:
        raise ValueError("Input list length must be divisible by 7")

    result = vals[:7]
    removed_frames = 0

    for i in range(7, len(vals), 7):
        current_first = vals[i]
        previous_first = result[-7]

        if current_first != previous_first:
            result.extend(vals[i:i+7])
        else:
            removed_frames += 1

    example[CODES_LIST_NAME] = result
    return example

def create_input_ids(example, tokenizer):
    # Map on dataset to create the final dataset with input_ids, labels, attention_mask
    text_prompt = f"{example['source']}: {example['text']}" if "source" in example else example["text"]

    text_ids = tokenizer.encode(text_prompt, add_special_tokens=True)
    text_ids.append(END_OF_TEXT)

    example["text_tokens"] = text_ids
    input_ids = (
        [START_OF_HUMAN]
        + example["text_tokens"]
        + [END_OF_HUMAN]
        + [START_OF_AI]
        + [START_OF_SPEECH]
        + example[CODES_LIST_NAME]
        + [END_OF_SPEECH]
        + [END_OF_AI]
    )
    example["input_ids"] = input_ids
    example["labels"] = input_ids
    example["attention_mask"] = [1] * len(input_ids)
    return example
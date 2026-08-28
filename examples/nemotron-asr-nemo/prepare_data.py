"""Prepare a NeMo ASR manifest for fine-tuning Nemotron 3.5 ASR.

This uses the small, public AN4 corpus as a stand-in for your own data so the
example runs end-to-end quickly. Swap in your own audio + transcripts by
producing JSON-lines manifests with the same schema:

    {"audio_filepath": "/abs/path/clip.wav", "duration": 4.27,
     "text": "reference transcript", "lang": "en-US", "target_lang": "en-US"}

Two details matter most for Nemotron 3.5 ASR:
  * Every clip carries a ``target_lang`` tag - this drives the model's
    prompt-based language conditioning. Use a locale the model recognizes
    (e.g. en-US, es-ES, el-GR, bg-BG).
  * Transcripts should match the base model's text style: punctuated and
    properly cased. AN4 is lowercase/unpunctuated, so expect it only as a
    smoke test, not a quality benchmark.
"""

import argparse
import glob
import json
import os
import subprocess
import tarfile
import urllib.request

AN4_URL = "https://dldata-public.s3.us-east-2.amazonaws.com/an4_sphere.tar.gz"


def download_and_extract_an4(data_dir: str) -> str:
    source_data_dir = os.path.join(data_dir, "an4")
    if os.path.exists(source_data_dir):
        print(f"AN4 already present at {source_data_dir}, skipping download.")
        return source_data_dir

    tar_path = os.path.join(data_dir, "an4_sphere.tar.gz")
    print(f"Downloading AN4 from {AN4_URL} ...")
    urllib.request.urlretrieve(AN4_URL, tar_path)
    print("Extracting AN4 ...")
    with tarfile.open(tar_path) as tar:
        tar.extractall(path=data_dir)
    return source_data_dir


def build_manifest(transcripts_path, manifest_path, target_wavs_dir, target_lang):
    import librosa

    with open(transcripts_path, "r") as fin, open(manifest_path, "w") as fout:
        for line in fin:
            # Lines look like: <s> transcript </s> (fileID)
            transcript = line[: line.find("(") - 1].lower()
            transcript = transcript.replace("<s>", "").replace("</s>", "").strip()

            file_id = line[line.find("(") + 1 : -2]
            audio_path = os.path.join(target_wavs_dir, file_id + ".wav")
            duration = librosa.core.get_duration(path=audio_path)

            metadata = {
                "audio_filepath": audio_path,
                "duration": duration,
                "text": transcript,
                "lang": target_lang,
                "target_lang": target_lang,
            }
            json.dump(metadata, fout)
            fout.write("\n")
    print(f"Wrote manifest: {manifest_path}")


def main(args):
    data_dir = args.data_dir
    os.makedirs(data_dir, exist_ok=True)

    source_data_dir = download_and_extract_an4(data_dir)
    target_data_dir = os.path.join(data_dir, "an4_converted")
    target_wavs_dir = os.path.join(target_data_dir, "wavs")
    os.makedirs(target_wavs_dir, exist_ok=True)

    # Convert the .sph source files to mono 16kHz .wav files.
    sph_list = glob.glob(os.path.join(source_data_dir, "**/*.sph"), recursive=True)
    print(f"Converting {len(sph_list)} .sph files to .wav ...")
    for sph_path in sph_list:
        wav_path = os.path.join(
            target_wavs_dir,
            os.path.splitext(os.path.basename(sph_path))[0] + ".wav",
        )
        subprocess.run(["sox", sph_path, "-r", "16000", "-c", "1", wav_path], check=True)

    build_manifest(
        os.path.join(source_data_dir, "etc/an4_train.transcription"),
        os.path.join(target_data_dir, "train_manifest.json"),
        target_wavs_dir,
        args.target_lang,
    )
    build_manifest(
        os.path.join(source_data_dir, "etc/an4_test.transcription"),
        os.path.join(target_data_dir, "test_manifest.json"),
        target_wavs_dir,
        args.target_lang,
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare AN4 manifests for NeMo ASR.")
    parser.add_argument(
        "--data_dir",
        type=str,
        default=os.environ.get("DATA_DIR", "./data"),
        help="Directory to download and process the dataset into.",
    )
    parser.add_argument(
        "--target_lang",
        type=str,
        default="en-US",
        help="Language tag applied to every clip (drives prompt conditioning).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())

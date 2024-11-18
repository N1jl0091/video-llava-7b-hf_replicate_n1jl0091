Video-llava-7b-hf_replicate

Process to upload model to replicate with quick model load time:
- Donwload weights from hugging face locally
- Add to directory containing other needed files
- Build docker image
- Push to replicate
- Faster Load Speeds!

Failed Methods of upload with quick boot:
- Replicate CDN: Only used for inside replicate, not for end users
- Alternative CDN: Expensive
- Hugging Face Transfer: Too Slow

The Model is Constantly being tweaked, but please only use the latest version - for the best results :)

NB: Weights are not included in this repo, due to large file size.

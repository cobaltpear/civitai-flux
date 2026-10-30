#!/bin/sh
# FLUX.1 Universal (any FLUX.1 model from Civitai or Hugging Face) — everything the worker image needs. The Dockerfile in your copy downloads this file
# from the original repo on every build, so fixes made there reach you with a rebuild.
set -eu
# civitai.env in your copy (made by Cobalt Pear's «Add your token» link) wins over the two Dockerfile lines
if [ -f /setup/civitai.env ]; then
  for k in CIVITAI_TOKEN CIVITAI_MODEL HF_TOKEN; do
    v=$(sed -n "s/^[[:space:]]*$k[[:space:]]*=//p" /setup/civitai.env | tr -d '\r"'"'"' ' | head -n 1)
    [ -z "$v" ] || eval "$k=\$v"
  done
fi
export CIVITAI_TOKEN CIVITAI_MODEL HF_TOKEN
case "${CIVITAI_MODEL:-}" in ""|PASTE_*) echo "ERROR: put the model in civitai.env (or at the top of the Dockerfile) — its Civitai page link or version number, or its Hugging Face link" >&2; exit 1;; esac

# Find the one file the link means, check it is a FLUX.1 model, then download it and check its SHA-256.
# Civitai: its API names the file. Hugging Face: the repo's single-file model (a .safetensors at its top, not the diffusers folders).
# The model is always saved as diffusion_models/model.safetensors — the pack's workflow loads that name.
python3 - <<'PY'
import hashlib, json, os, re, struct, sys, urllib.request
token, link, hf_token = os.environ.get("CIVITAI_TOKEN", ""), os.environ["CIVITAI_MODEL"].strip(), os.environ.get("HF_TOKEN", "")
def fail(msg): sys.exit("ERROR: " + msg)
def download(url, dest, sha, name, headers={}):
    h = hashlib.sha256()
    with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "cobaltpear", **headers}), timeout=600) as r, open(dest, "wb") as out:
        while chunk := r.read(1 << 22): h.update(chunk); out.write(chunk)
    if sha and h.hexdigest() != sha: fail(f"{name} does not match its SHA-256 — try the build again")
    print("SHA-256 OK" if sha else "No SHA-256 listed for this file — not checked")
dest = "/comfyui/models/diffusion_models/model.safetensors"
hf = None if "civitai.com" in link or re.fullmatch(r"\d+", link) else \
    re.fullmatch(r"(?:https?://huggingface\.co/)?([\w.-]+/[\w.-]+)(?:/(?:blob|resolve|tree)/[^/]+(?:/(.+?))?)?/?(?:\?.*)?", link)
if hf:
    repo, want = hf.group(1), hf.group(2)
    auth = {"Authorization": "Bearer " + hf_token} if hf_token and not hf_token.startswith("PASTE_") else {}
    def get(url, **kw):
        try: return urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "cobaltpear", **auth, **kw}), timeout=60)
        except urllib.error.HTTPError as e:
            fail(f"Hugging Face answered {e.code} for {repo}" + (" — it is gated: accept its license on its page and put HF_TOKEN=<a Read token> in civitai.env" if e.code in (401, 403) else ""))
    info = json.load(get(f"https://huggingface.co/api/models/{repo}?blobs=true"))
    files = [f for f in info.get("siblings", []) if f["rfilename"].endswith(".safetensors") and (f["rfilename"] == want if want else "/" not in f["rfilename"])]
    files = [f for f in files if f.get("size", 0) > 5e+09 or want]  # a whole model, not a VAE or a LoRA
    if not files: fail(f"{repo} has no single-file checkpoint (.safetensors) at its top — it only has diffusers folders, which this worker can't load")
    f = min(files, key=lambda f: f.get("size", 0))  # fp16 rather than fp32
    url = f"https://huggingface.co/{repo}/resolve/main/{f['rfilename']}"
    head = get(url, Range="bytes=0-7").read(); n = struct.unpack("<Q", head)[0]
    keys = list(json.loads(get(url, Range=f"bytes=8-{7 + n}").read()))
    if not (any(k.endswith("double_blocks.0.img_attn.qkv.weight") for k in keys) and not any("distilled_guidance_layer" in k for k in keys)): fail(f"{f['rfilename']} is not a FLUX.1 model — this worker takes FLUX.1 (dev, schnell, Krea) models only")
    print(f"Hugging Face: {repo} — {f['rfilename']}")
    print(f"Downloading {f['rfilename']} ({f.get('size', 0) / 1e9:.1f} GB)")
    download(url, dest, (f.get("lfs") or {}).get("sha256", ""), f["rfilename"], auth)
    json.dump({"repo": repo, "name": repo.split("/")[-1], "file": f["rfilename"], "base": "FLUX.1"}, open("/comfyui/models/diffusion_models/model.json", "w"))
    sys.exit(0)
if not token or token.startswith("PASTE_"): fail("put your Civitai token in civitai.env (or at the top of the Dockerfile)")
# a model file ComfyUI loads as is: .safetensors, not NF4 / NVFP4 / INT8 (those need extra nodes)
def usable(v):
    return [f for f in v.get("files", []) if f.get("type") in ("Model", "Diffusion Model") and f.get("metadata", {}).get("format") == "SafeTensor"
            and f.get("metadata", {}).get("fp") not in ("nf4", "int8") and not re.search(r"nvfp4|nf4|int8|convrot", f["name"], re.I)]
def get(path):
    req = urllib.request.Request("https://civitai.com/api/v1/" + path, headers={"Authorization": "Bearer " + token, "User-Agent": "cobaltpear"})
    try: return json.load(urllib.request.urlopen(req, timeout=60))
    except Exception as e: fail(f"could not ask Civitai about {link!r} ({e})")
m = re.search(r"modelVersionId=(\d+)", link) or re.fullmatch(r"(\d+)", link)
if m: version = get(f"model-versions/{m.group(1)}")
else:
    m = re.search(r"/models/(\d+)", link) or fail(f"CIVITAI_MODEL {link!r} is not a Civitai model link")
    vs = get(f"models/{m.group(1)}")["modelVersions"]
    version = next((v for v in vs if usable(v)), vs[0])   # the newest version with a file this worker can load
    version = get(f"model-versions/{version['id']}")
base, kind = version.get("baseModel", ""), version.get("model", {}).get("type", "")
print(f"Civitai: {version.get('model', {}).get('name')} — {version.get('name')} (version {version['id']}, {base}, {kind})")
if kind != "Checkpoint": fail(f"this is a {kind}, not a Checkpoint")
if base not in ['Flux.1 D', 'Flux.1 S', 'Flux.1 Krea']: fail(f"this is a {base} model — this worker takes FLUX.1 (dev, schnell, Krea) models (Flux.1 D, Flux.1 S, Flux.1 Krea on Civitai)")
files = usable(version)
if not files: fail("no .safetensors model file in this version that plain ComfyUI can load (GGUF, NF4, NVFP4 and INT8 files are not taken) — pick another version")
f = min(files, key=lambda f: f.get("sizeKB", 0))  # fp8 rather than fp16: faster to build, fits a 24 GB GPU
print(f"Downloading {f['name']} ({f.get('sizeKB', 0) / 1e6:.1f} GB)")
download(f["downloadUrl"] + ("&" if "?" in f["downloadUrl"] else "?") + "token=" + token, dest, f.get("hashes", {}).get("SHA256", "").lower(), f["name"])
json.dump({"version": version["id"], "name": version.get("model", {}).get("name"), "file": f["name"], "base": base}, open("/comfyui/models/diffusion_models/model.json", "w"))
def local_keys(p):
    with open(p, "rb") as fh: n = struct.unpack("<Q", fh.read(8))[0]; return list(json.loads(fh.read(n)))
keys = local_keys(dest)
if not (any(k.endswith("double_blocks.0.img_attn.qkv.weight") for k in keys) and not any("distilled_guidance_layer" in k for k in keys)): fail(f"{f['name']} is not a FLUX.1 model inside, though Civitai lists it as {base} — pick another version")
PY

# the text encoders and VAE every FLUX.1 model uses, pinned by SHA-256
file() {
  wget -q -O "$1" "$2"
  echo "$3  $1" | sha256sum -c - || { echo "ERROR: $1 changed upstream — wait for an update of the original repo, then rebuild" >&2; exit 1; }
}

file /comfyui/models/text_encoders/t5xxl_fp8_e4m3fn.safetensors \
  "https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/6af2a98e3f615bdfa612fbd85da93d1ed5f69ef5/t5xxl_fp8_e4m3fn.safetensors" \
  7d330da4816157540d6bb7838bf63a0f02f573fc48ca4d8de34bb0cbfd514f09

file /comfyui/models/text_encoders/clip_l.safetensors \
  "https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/6af2a98e3f615bdfa612fbd85da93d1ed5f69ef5/clip_l.safetensors" \
  660c6f5b1abae9dc498ac2d21e1347d2abdb0cf6c0c0c8576cd796491d9a6cdd

file /comfyui/models/vae/ae.safetensors \
  "https://huggingface.co/Comfy-Org/Lumina-Image-2/resolve/5b072540ef86570fecb8249c505f23d5bdeb88cd/split_files/vae/ae.safetensors" \
  afc8e28272cd15db3919bacdb6918ce9c1ed22e96cb12c4d5ed0fba823529e38

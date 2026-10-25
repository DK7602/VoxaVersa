# Training the AI voices on a free Google Colab GPU

This runs VoxaVersa's full voice-training queue on one of Google's GPUs instead of a CPU. It is the same
trainer, the same data, the same quality checks and the same automatic choice of the best checkpoint. It is
just faster: about 1 - 3 hours per singer instead of 14 - 45.

| | What it is |
|---|---|
| Notebook | `tools/training/VoxaVersa_Queue_Colab.ipynb` |
| Data kit | `VoxaVersa_Colab_Data.zip` (private: the cleaned Cymatics singers + test vocals + the voice list; never share it) |
| Results | `.vxpack` voice packs in *My Drive > VoxaVersa > queue > voices* |

## 1. One-time preparation (5 minutes)

1. Sign in to [Google Drive](https://drive.google.com) with your Google account.
2. Create a folder named **VoxaVersa** in *My Drive* (New > New folder).
3. Upload **VoxaVersa_Colab_Data.zip** into it, without unzipping it (New > File upload). It is ~185 MB.
   The path must be *My Drive > VoxaVersa > VoxaVersa_Colab_Data.zip*.
4. Open the notebook from GitHub, so it is always the newest version: in
   [Colab](https://colab.research.google.com) choose *File > Open notebook*, the **GitHub** tab, type
   `DK7602/VoxaVersa`, pick the branch `claude/voxaversa-voice-plugin-20arm6` and click
   `tools/training/VoxaVersa_Queue_Colab.ipynb`. (Or *File > Upload notebook* with the `.ipynb` file. The
   notebook fetches the newest VoxaVersa tools by itself either way.)

## 2. Training (repeat until all voices are done)

1. Open the notebook in Colab.
2. Menu **Runtime > Change runtime type**, choose **T4 GPU**, **Save**.
3. Menu **Runtime > Run all**. When asked, allow access to your Google Drive (choose your account, tick
   "See, edit, create and delete all of your Google Drive files": the notebook only uses the VoxaVersa
   folder).
4. Wait for step 6. The first time, steps 3 - 4 take about 5 - 10 minutes (software and base models). Step 5
   takes seconds: training starts with the Cymatics singers while the 20 free VocalSet singers download and are
   cleaned in the background (~30 - 60 minutes; the line "VocalSet singers:" in step 6 shows how far it is).
   They join the queue by themselves when ready. If that download fails, training simply carries on and the
   next Run all tries again. Later sessions start in about 5 minutes.
5. Step 6 refreshes every 2 minutes: which singer is training, its epoch, the speed, and the time left.
   Leave the browser tab open.

**When Colab disconnects** (free accounts get a few GPU hours a day, and Colab stops idle sessions): nothing is
lost except the minutes since the last checkpoint (saved to Drive every 5 epochs, and before every quality
check). Come back later, open the notebook and do **Runtime > Run all** again. It continues where it stopped.

**What to expect:** 24 singers (4 from your Cymatics packs, 20 from VocalSet, a free CC BY 4.0 dataset).
Each trains for at least 100 and at most 300 epochs, stops by itself when the quality stops improving, and
keeps its best checkpoint. On a T4 with mixed precision that is roughly 1 - 3 hours per singer, so about
40 - 60 GPU hours in total: a couple of weeks of daily sessions on the free plan, or a few days with
Colab Pro (paid; check its current price). The order is female singers first.

**Settings** (step 2 of the notebook; the defaults are right for most people):

| Setting | Default | Meaning |
|---|---|---|
| `INCLUDE_VOCALSET` | on | Also train the 20 VocalSet singers. Off: only the 4 Cymatics singers (~6 GPU hours). |
| `PRECISION` | mixed | RVC's standard GPU training (fp16 mixed precision). *full* = fp32 like the CPU trainer, ~2x slower. |
| `BATCH_SIZE` | 8 | RVC's usual GPU batch. Out of GPU memory: the trainer halves it by itself. |

**Google Drive space:** about 5 - 8 GB while training (VocalSet 1.7 GB, the singer in training ~1.7 GB,
~0.15 GB per finished singer). Finished singers are cleaned up automatically, so the voice pack and the chosen
checkpoint are all that stay.

## 3. Installing the voices in VoxaVersa

Once (on your computer):

1. **ONNX Runtime**, the engine VoxaVersa uses for AI voices: see [NEURAL_ENGINE.md, section 3](NEURAL_ENGINE.md#3-installing-onnx-runtime).
2. **The content encoder:** download `contentvec-768.onnx` from *My Drive > VoxaVersa > queue* and put it in
   VoxaVersa's `Encoders` folder:
   * Windows: `%APPDATA%\VoxaVersa\Encoders\`
   * macOS: `~/Library/Application Support/VoxaVersa/Encoders/`
   * Linux: `~/.config/VoxaVersa/Encoders/`

For each finished singer:

3. Download its `.vxpack` from *My Drive > VoxaVersa > queue > voices > (singer)*.
4. In VoxaVersa: menu (top right) > **Import Voice Pack...** and pick the `.vxpack`.

Each pack already lists the built-in genre voices it powers (for example Female Pop and Female EDM). Those
voices show an **AI** badge and use the trained singer whenever *Use AI Voice Engine* is on. Packs that only
power built-in voices don't appear as voices of their own. When more singers finish, the assignment can
change: import the newest `.vxpack` files again (they replace the old ones).

The notebook's step 7 plays a before/after sample for every finished singer and shows which built-in voices
it powers.

## 4. If something goes wrong

| Message | What to do |
|---|---|
| "No GPU" | Runtime > Change runtime type > T4 GPU > Save, then Run all. If no GPU is available today, try later. |
| "Not found in your Drive" | The zip must be at *My Drive > VoxaVersa > VoxaVersa_Colab_Data.zip* (exact names). |
| "Step failed: uv pip install ..." | Run all again (a download hiccup). If it repeats, copy the message for support. |
| Step 6 says *supervisor: not running* and shows an error | Copy the last lines of the log; Run all once more (it retries). |
| Colab: "You cannot currently connect to a GPU" | The daily free GPU time is used up. Come back later or tomorrow. |

Technical details: `tools/training/colab_queue.py` (setup, data, queue file), `train_queue.py` (the queue;
`"device": "cuda"`), `fetch_vocalset.py` (VocalSet download, bit-identical to the CPU queue's data) and
[TRAINING.md](TRAINING.md).

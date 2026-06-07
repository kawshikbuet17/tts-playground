export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=1

CUDA_VISIBLE_DEVICES=1 python test_omnivoice_full_benchmark_batch.py \
  --suite all \
  --model k2-fsa/OmniVoice \
  --ref-audio ref_audio.wav \
  --ref-text "বসের নির্দেশ, অন্য কাগজের কাছে খবরটা যাওয়ার আগেই আমাদেরকে কোনোভাবে স্পটে পৌছাতে হবে। অগত্যা তাই যেতেই হলো।" \
  --instruct "female, middle-aged, moderate pitch, indian accent" \
  --dtype bf16 \
  --num-step 8 \
  --speed 1.0 \
  --benchmark-kind all \
  --batch-size 4 \
  --batch-duration 100.0 \
  --audio-chunk-duration 30.0 \
  --audio-chunk-threshold 60.0 \
  --output-dir outputs/omnivoice_full_benchmark_all_step8_batch_compare
# 1) 单图，只保留一个点（默认），保存叠加图 + 修复图 + 掩膜 + CSV
python detect_moon_glint.py \
  --input /path/to/keogram.jpg \
  --output ./out \
  --save-overlay --save-inpaint --save-mask \
  --csv ./out/detections.csv

# 2) 批量处理文件夹，保留 top-1（或改为 top-2/3），自动导出 CSV
python detect_moon_glint.py \
  --input /path/to/folder \
  --output ./out_batch \
  --k 1 --save-overlay --save-mask --save-inpaint \
  --csv ./out_batch/detections.csv
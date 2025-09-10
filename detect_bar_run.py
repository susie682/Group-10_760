rows2 = run_detect(
    input_path="./2015",
    output_dir="./keogram-out2",
    save_overlay=True, save_mask=True, save_inpaint=True,
    csv_path="./keogram-out2/detections.csv"
)
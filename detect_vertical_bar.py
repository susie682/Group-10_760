def detect_vertical_white_purple_bars(img_bgr: np.ndarray) -> np.ndarray:

    H, W = img_bgr.shape[:2]
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).astype(np.int32)
    Hc, S, V = cv2.split(hsv)

    # 白色门控
    white  = (S <= 70) & (V >= 210)
    # 紫色门控
    purple = (Hc >= 135) & (Hc <= 175) & (S >= 70) & (V >= 120)
    cand = (white | purple).astype(np.uint8)

    # 每列占比 + 平滑
    col_frac = cand.sum(axis=0) / float(H)
    kernel = np.ones(21, np.float32)/21
    col_frac = np.convolve(col_frac, kernel, mode="same")
    cols = (col_frac >= 0.35).astype(np.uint8)

    # 合并连续列 -> 掩膜
    bar_mask = np.zeros((H,W), np.uint8)
    in_bar, start = False, 0
    for x in range(W):
        if cols[x] and not in_bar:
            in_bar, start = True, x
        if (not cols[x] or x==W-1) and in_bar:
            end = x if not cols[x] else x
            w = end - start + 1
            if w >= 4 and w <= 0.2*W:  # 合理宽度
                bar_mask[:, start:end+1] = 255
            in_bar = False

    # 纵向覆盖过滤
    num, labels, stats, _ = cv2.connectedComponentsWithStats(bar_mask, connectivity=8)
    left_candidates, right_candidates = [], []
    for i in range(1, num):
        x,y,w,h,area = stats[i]
        if h/float(H) >= 0.85 and h/max(w,1) >= 6:
            cx = x+w/2
            if cx <= 0.5*W:
                left_candidates.append((area,i))
            else:
                right_candidates.append((area,i))

    # 保留左右各一个（面积最大）
    final = np.zeros_like(bar_mask)
    if left_candidates:
        _, idx = max(left_candidates, key=lambda t:t[0])
        final[labels==idx] = 255
    if right_candidates:
        _, idx = max(right_candidates, key=lambda t:t[0])
        final[labels==idx] = 255

    return final

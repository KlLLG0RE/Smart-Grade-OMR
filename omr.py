import cv2
import numpy as np


def process_omr_bubbles(image_bytes, answer_key, choices_per_question=4):

    np_arr = np.frombuffer(image_bytes, np.uint8)
    image  = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if image is None:
        return {"error": "Could not decode image."}

    img_h, img_w = image.shape[:2]
    print(f"[DEBUG] Image size: {img_w}x{img_h}")

    gray    = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    _, otsu = cv2.threshold(blurred, 0, 255,
                             cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

    adaptive = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        21, 2
    )

    contours, _ = cv2.findContours(adaptive.copy(),
                                   cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

    scale    = img_w / 415.0
    min_w    = int(48  * scale)
    max_w    = int(80  * scale)
    min_h    = int(22  * scale)
    max_h    = int(42  * scale)
    min_area = int(250 * scale * scale)

    centres = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        ar   = w / float(h)
        area = cv2.contourArea(c)
        if (min_w <= w <= max_w and
                min_h <= h <= max_h and
                1.4 <= ar <= 2.8 and
                area >= min_area):
            centres.append((x + w // 2, y + h // 2))

    print(f"[DEBUG] Bubble centres found: {len(centres)}")

    if len(centres) < choices_per_question:
        return {"error": "Could not detect enough bubbles. Make sure the sheet is flat, well-lit, and camera is directly above."}

    xs = sorted(set(cx for cx, _ in centres))
    cols = []
    for x in xs:
        if not cols or x - cols[-1] > int(30 * scale):
            cols.append(x)
        else:
            cols[-1] = (cols[-1] + x) // 2

    cols = cols[:choices_per_question]

    if len(cols) < choices_per_question:
        return {"error": f"Only {len(cols)} columns detected, expected {choices_per_question}."}

    print(f"[DEBUG] Columns: {cols}")

    ys = sorted(set(cy for _, cy in centres))
    rows = []
    for y in ys:
        if not rows or y - rows[-1] > int(30 * scale):
            rows.append(y)
        else:
            rows[-1] = (rows[-1] + y) // 2

    if len(rows) >= 2:
        spacings = [rows[i + 1] - rows[i] for i in range(len(rows) - 1)]
        spacing  = int(np.median(spacings))
        full_rows = [rows[0]]
        for r in rows[1:]:
            while r - full_rows[-1] > spacing * 1.4:
                full_rows.append(full_rows[-1] + spacing)
            full_rows.append(r)
        rows = full_rows

    print(f"[DEBUG] Rows: {rows}")

    num_questions = len(answer_key)
    rows = rows[:num_questions]

    hw = int(35 * scale)
    hh = int(16 * scale)

    correct_answers = 0
    student_answers = []
    options         = ["A", "B", "C", "D", "E"]

    for qi, ry in enumerate(rows):
        if qi >= num_questions:
            break

        pixel_counts = []
        for cx in cols:
            x1 = max(0, cx - hw)
            x2 = min(img_w, cx + hw)
            y1 = max(0, ry - hh)
            y2 = min(img_h, ry + hh)
            patch   = otsu[y1:y2, x1:x2]
            dark_px = cv2.countNonZero(patch)
            pixel_counts.append(dark_px)

        best_idx = pixel_counts.index(max(pixel_counts))
        student_answers.append(best_idx)

        print(f"[DEBUG] Q{qi+1}: detected={options[best_idx]}, key={answer_key.get(qi, -1)}, counts={pixel_counts}")

        color = (0, 0, 255)
        if best_idx == answer_key.get(qi, -1):
            color = (0, 255, 0)
            correct_answers += 1

        detected_cx = cols[best_idx]
        radius = int(hw * 0.85)
        cv2.circle(image, (detected_cx, ry), radius, color, 3)

    if num_questions == 0:
        return {"error": "Answer key is empty."}

    score = (correct_answers / num_questions) * 100
    cv2.imwrite("static/result.jpg", image)

    return {
        "score":            round(score, 2),
        "student_answers":  student_answers,
        "correct":          correct_answers,
        "total":            num_questions,
        "result_image_url": "/static/result.jpg"
    }

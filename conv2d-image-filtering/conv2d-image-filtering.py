def conv2d(image, kernel, stride=1, padding=0):
    """
    Apply 2D convolution to a single-channel image.
    """
    # Write code here
    img_h = len(image)
    img_w = len(image[0])

    padded_w = img_w + 2 * padding
    padded_h = img_h + 2 * padding

    padded_img = [[0.0 for j in range(padded_w)] for i in range(padded_h)]

    for r in range(img_h):
        for c in range(img_w):
            padded_img[r+padding][c+padding] = image[r][c]

    kernel_h = len(kernel)
    kernel_w = len(kernel[0])

    out_h = (img_h + 2 * padding - kernel_h) // stride + 1
    out_w = (img_w + 2 * padding - kernel_w) // stride + 1

    output_img = [[0.0 for _ in range(out_w)] for _ in range(out_h)]


    for i in range(out_h):
        for j in range(out_w):
            start_row = i * stride
            start_col = j * stride

            curr_sum = 0
            for m in range(kernel_h):
                for n in range(kernel_w):
                    if start_row + m < len(padded_img) and start_col + n < len(padded_img[0]):
                        curr_sum += padded_img[start_row+m][start_col+n] * kernel[m][n]
            output_img[i][j] = curr_sum

    return output_img
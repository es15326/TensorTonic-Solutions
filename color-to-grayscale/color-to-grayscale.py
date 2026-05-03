def color_to_grayscale(image):
    """
    Convert an RGB image to grayscale using luminance weights.
    """
    # Write code here
    img_h = len(image)
    img_w = len(image[0])

    grayscale_img = [[0.0 for _ in range(img_w)] for _ in range(img_h)]

    for r in range(img_h):
        for c in range(img_w):
            red = image[r][c][0]
            green = image[r][c][1]
            blue = image[r][c][2]

            grayscale_img[r][c] = 0.299 * red + 0.587 * green + 0.114 * blue

    return grayscale_img
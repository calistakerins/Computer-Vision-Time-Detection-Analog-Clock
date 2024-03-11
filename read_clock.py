import cv2
import numpy as np

# Load template image of a clock face
template = cv2.imread('images/clock_template.png', 0)

# Load image
image = cv2.imread('images/clock.png')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Gaussian blur
blur = cv2.GaussianBlur(gray, (5, 5), 0)

# Apply edge detection
edges = cv2.Canny(blur, 50, 150)

# Hough Circle Transform
circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, dp=1, minDist=20, param1=50, param2=30, minRadius=0, maxRadius=0)

# Template matching threshold
threshold = 0.8

if circles is not None:
    circles = np.uint16(np.around(circles))

    for circle in circles[0, :]:
        center = (circle[0], circle[1])
        radius = circle[2]

        # Extract region of interest
        # Convert image data to np.int32 before creating ROI to prevent overflow
        roi = gray[np.int32(circle[1])-radius:np.int32(circle[1])+radius, np.int32(circle[0])-radius:np.int32(circle[0])+radius]

        # Resize template to match size of roi
        new_width = roi.shape[1]
        new_height = roi.shape[0]
        print(new_width)
        print(new_height)
        template_resized = cv2.resize(template, (new_width, new_height))

        # Perform template matching
        result = cv2.matchTemplate(roi, template_resized, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(result)

        # If maximum correlation value exceeds threshold, consider it a clock
        if max_val > threshold:
            # Draw circle
            cv2.circle(image, center, radius, (0, 255, 0), 2)
            break


# Display result
cv2.imshow('Clock Detection', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

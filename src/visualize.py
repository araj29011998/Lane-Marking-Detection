import matplotlib.pyplot as plt
import cv2

def visualize(image, lines, original_image):
    # Draw lines on the original image
    line_image = cv2.addWeighted(original_image, 0.8, lines, 1, 1)
    
    # Display result
    plt.imshow(cv2.cvtColor(line_image, cv2.COLOR_BGR2RGB))
    plt.show()
